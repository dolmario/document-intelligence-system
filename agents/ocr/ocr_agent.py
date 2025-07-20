import asyncio
import base64
import json
import logging
import os
import io
import gc
import asyncpg
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pdf2image
from typing import List, Dict, Optional
from datetime import datetime

# Optional imports with fallbacks
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    logging.warning("python-docx not available - DOCX support disabled")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logging.warning("pandas not available - Excel/CSV support limited")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enhanced_ocr_agent")

class EnhancedOCRAgent:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.db_pool = None
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.running = True
        
        # Streaming settings
        self.MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB threshold für streaming
        
        # OCR Configuration
        self.tesseract_config = '--oem 3 --psm 6'
        self.tesseract_config_fast = '--oem 1 --psm 6'  # Schnellere Engine für große Dateien

    async def connect(self):
        self.db_pool = await asyncpg.create_pool(self.db_url)
        logger.info("Connected to PostgreSQL")

    async def close(self):
        if self.db_pool:
            await self.db_pool.close()

    async def process_queue(self):
        while self.running:
            try:
                async with self.db_pool.acquire() as conn:
                    task = await conn.fetchrow("""
                        UPDATE processing_queue 
                        SET status = 'processing', started_at = NOW()
                        WHERE id = (
                            SELECT id FROM processing_queue 
                            WHERE status = 'pending' 
                            AND task_type IN ('ocr', 'extract')
                            ORDER BY priority DESC, created_at ASC
                            LIMIT 1
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING *
                    """)
                    if task:
                        await self.process_document(task)
                    else:
                        await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(10)

    async def process_document(self, task):
        async with self.db_pool.acquire() as conn:
            try:
                doc = await conn.fetchrow("SELECT * FROM documents WHERE id = $1", task["document_id"])
                if not doc:
                    raise ValueError(f"Document {task['document_id']} not found")

                chunks = await self.extract_content(doc)

                # Save all chunks
                for idx, chunk in enumerate(chunks):
                    await conn.execute("""
                        INSERT INTO chunks (document_id, chunk_index, content, content_type, page_number, position, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (document_id, chunk_index) DO UPDATE
                        SET content = EXCLUDED.content, content_type = EXCLUDED.content_type, metadata = EXCLUDED.metadata
                    """, doc["id"], idx, chunk["content"], chunk["type"], 
                        chunk.get("page"), chunk.get("position"), json.dumps(chunk.get("metadata", {})))

                await conn.execute("UPDATE documents SET status = 'completed', processed_at = NOW() WHERE id = $1", doc["id"])
                await conn.execute("UPDATE processing_queue SET status = 'completed', completed_at = NOW() WHERE id = $1", task["id"])

                # Create enhancement task
                await conn.execute("""
                    INSERT INTO processing_queue (document_id, task_type, priority)
                    VALUES ($1, 'enhance', $2)
                """, doc["id"], task["priority"])

                logger.info(f"Processed document {doc['original_name']}: {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Document processing error: {e}")
                await conn.execute("UPDATE processing_queue SET status = 'failed', error_message = $2, completed_at = NOW() WHERE id = $1", task["id"], str(e))
                await conn.execute("UPDATE documents SET status = 'failed' WHERE id = $1", task["document_id"])

    async def extract_content(self, doc: Dict) -> List[Dict]:
        """Enhanced content extraction with streaming support"""
        file_type = doc['file_type'].lower()
        metadata = doc.get('metadata', {})

        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        file_content = metadata.get('file_content')
        file_path = metadata.get('file_path')
        processing_mode = metadata.get('processing_mode', 'in_memory')

        logger.info(f"Processing mode: {processing_mode}, file_type: {file_type}")

        # Streaming mode for large files
        if processing_mode == 'streaming' and file_path and os.path.exists(file_path):
            return await self._extract_content_streaming(file_path, file_type, doc['original_name'])
        
        # In-memory mode for smaller files
        if file_content:
            try:
                file_bytes = base64.b64decode(file_content.encode("utf-8"))
            except Exception as e:
                logger.error(f"Base64 decode error: {e}")
                raise ValueError("Invalid base64 content")
        elif file_path and os.path.exists(file_path):
            # Fallback: read file normally
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
        else:
            raise ValueError("No file content available")

        # Extract based on type
        if file_type == "pdf":
            return await self.extract_pdf(file_bytes, doc['original_name'])
        elif file_type in ["png", "jpg", "jpeg", "tiff", "tif"]:
            return await self.extract_image(file_bytes)
        elif file_type in ["txt", "md", "log"]:
            return self.extract_text(file_bytes.decode("utf-8", errors="ignore"))
        elif file_type in ["docx", "doc"]:
            if HAS_DOCX:
                return await self.extract_docx(file_bytes)
            else:
                return self.extract_text(file_bytes.decode("utf-8", errors="ignore"))
        else:
            return self.extract_text(file_bytes.decode("utf-8", errors="ignore"))

    async def _extract_content_streaming(self, file_path: str, file_type: str, filename: str) -> List[Dict]:
        """Streaming extraction for large files"""
        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            logger.info(f"Streaming large file: {filename} ({file_size:.1f}MB)")
            
            if file_type == 'pdf':
                return await self._stream_large_pdf(file_path)
            elif file_type in ['txt', 'md', 'log']:
                return await self._stream_large_text(file_path)
            else:
                # Fallback for other file types
                logger.warning(f"Streaming fallback for {file_type}")
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                return self.extract_text(file_bytes.decode('utf-8', errors='ignore'))
                
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            # Ultimate fallback
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            return self.extract_text(file_bytes.decode('utf-8', errors='ignore'))

    async def _stream_large_pdf(self, file_path: str) -> List[Dict]:
        """Verarbeite PDF seitenweise mit dynamischer DPI"""
        chunks = []
        try:
            # Dynamische DPI basierend auf Dateigröße
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            dpi = 150 if file_size > 50 else 300  # Höhere DPI nur bei kleinen PDFs
            
            # Tesseract config für große Dateien optimieren
            tesseract_config = self.tesseract_config_fast if file_size > 100 else self.tesseract_config
            
            logger.info(f"PDF streaming: {file_size:.1f}MB, DPI: {dpi}")
            
            # Seitenweise laden
            images = pdf2image.convert_from_path(file_path, dpi=dpi)
            total_pages = len(images)
            
            for page_num, image in enumerate(images, 1):
                # Progress für sehr große PDFs
                if total_pages > 50 and page_num % 10 == 0:
                    logger.info(f"📄 Fortschritt: {page_num}/{total_pages} ({page_num/total_pages*100:.1f}%)")
                elif total_pages <= 50:
                    logger.info(f"Verarbeite Seite {page_num}/{total_pages}")
                
                # OCR pro Seite
                text = pytesseract.image_to_string(
                    image, 
                    lang=os.getenv("TESSERACT_LANG", "deu+eng"),
                    config=tesseract_config
                )
                
                if text.strip():
                    page_chunks = self.split_text(text)
                    for chunk_idx, chunk in enumerate(page_chunks):
                        chunks.append({
                            "content": chunk,
                            "type": "text",
                            "page": page_num,
                            "metadata": {
                                "extraction_method": "pdf_streaming",
                                "dpi": dpi,
                                "page_chunk": chunk_idx
                            }
                        })
                
                # Speicher sparen
                del image
                if page_num % 5 == 0:  # Alle 5 Seiten explizit freigeben
                    gc.collect()
                    
        except Exception as e:
            logger.error(f"PDF-Streaming fehlgeschlagen: {e}")
            raise
        
        logger.info(f"PDF streaming completed: {len(chunks)} chunks from {total_pages} pages")
        return chunks

    async def _stream_large_text(self, file_path: str) -> List[Dict]:
        """Verarbeite Textdatei line-by-line mit Fehlerbehandlung"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Dynamische chunk size für große Dateien
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        dynamic_chunk_size = min(self.chunk_size * 2, 2000) if file_size > 100 else self.chunk_size
        
        logger.info(f"Text streaming: {file_size:.1f}MB, chunk_size: {dynamic_chunk_size}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = 0
                for line in f:
                    line = line.strip() + "\n"
                    if not line.strip():
                        continue  # Leere Zeilen überspringen
                    
                    current_chunk.append(line)
                    current_size += len(line)
                    line_count += 1
                    
                    if current_size >= dynamic_chunk_size:
                        text = ''.join(current_chunk)
                        chunks.append({
                            "content": text, 
                            "type": "text",
                            "metadata": {
                                "extraction_method": "text_streaming",
                                "lines_in_chunk": len(current_chunk),
                                "chunk_index": len(chunks)
                            }
                        })
                        current_chunk = []
                        current_size = 0
                        
                        if len(chunks) % 10 == 0:
                            logger.info(f"Processed {len(chunks)} chunks ({line_count} lines)")
                            
        except Exception as e:
            logger.error(f"Fehler beim Text-Streaming: {e}")
            raise
        
        # Rest abschließen
        if current_chunk:
            chunks.append({
                "content": ''.join(current_chunk), 
                "type": "text",
                "metadata": {
                    "extraction_method": "text_streaming",
                    "lines_in_chunk": len(current_chunk),
                    "chunk_index": len(chunks),
                    "final_chunk": True
                }
            })
        
        logger.info(f"Text streaming completed: {len(chunks)} chunks")
        return chunks

    async def extract_pdf(self, pdf_bytes: bytes, filename: str) -> List[Dict]:
        """Standard PDF extraction for smaller files"""
        chunks = []
        try:
            images = pdf2image.convert_from_bytes(pdf_bytes, dpi=300)
            logger.info(f"PDF in-memory processing: {filename}, {len(images)} pages")
            
            for page_num, image in enumerate(images, 1):
                text = pytesseract.image_to_string(
                    image, 
                    lang=os.getenv('TESSERACT_LANG', 'deu+eng'),
                    config=self.tesseract_config
                )
                
                if text.strip():
                    page_chunks = self.split_text(text)
                    for chunk in page_chunks:
                        chunks.append({
                            'content': chunk,
                            'type': 'text',
                            'page': page_num,
                            'metadata': {
                                'extraction_method': 'pdf_in_memory'
                            }
                        })
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return [{"content": f"[PDF extraction failed: {e}]", "type": "error"}]
        
        return chunks

    async def extract_image(self, image_bytes: bytes) -> List[Dict]:
        """Extract text from image"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(
                image,
                lang=os.getenv('TESSERACT_LANG', 'deu+eng'),
                config=self.tesseract_config
            )
            
            chunks = self.split_text(text) if text.strip() else ["[Bild ohne erkannten Text]"]
            return [{'content': chunk, 'type': 'text', 'page': 1} for chunk in chunks]
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            return [{"content": f"[Image extraction failed: {e}]", "type": "error"}]

    async def extract_docx(self, docx_bytes: bytes) -> List[Dict]:
        """Extract content from DOCX"""
        if not HAS_DOCX:
            return [{'content': '[DOCX file - python-docx not installed]', 'type': 'text'}]
            
        try:
            chunks = []
            doc = DocxDocument(io.BytesIO(docx_bytes))
            
            full_text = '\n'.join([para.text for para in doc.paragraphs])
            text_chunks = self.split_text(full_text)
            
            for chunk in text_chunks:
                chunks.append({'content': chunk, 'type': 'text'})
            
            # Extract tables
            for table_idx, table in enumerate(doc.tables):
                table_data = []
                for row in table.rows:
                    table_data.append([cell.text for cell in row.cells])
                
                if table_data:
                    chunks.append({
                        'content': json.dumps(table_data),
                        'type': 'table',
                        'metadata': {'table_index': table_idx}
                    })
            
            return chunks
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return [{"content": f"[DOCX extraction failed: {e}]", "type": "error"}]

    def extract_text(self, text: str) -> List[Dict]:
        """Extract plain text"""
        chunks = self.split_text(text)
        return [{"content": chunk, "type": "text"} for chunk in chunks]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks or [""]

    async def run(self):
        await self.connect()
        logger.info("Enhanced OCR Agent with Streaming started")
        try:
            await self.process_queue()
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            await self.close()


def main():
    db_url = os.getenv("DATABASE_URL", "postgresql://semanticuser:semantic2024@postgres:5432/semantic_doc_finder")
    agent = EnhancedOCRAgent(db_url)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
