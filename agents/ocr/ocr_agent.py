import asyncio
import base64
import json
import logging
import os
import io
import asyncpg
import pytesseract
from PIL import Image
import pdf2image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ocr_agent")

class OCRAgentV2:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.db_pool = None
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.running = True

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

                for idx, chunk in enumerate(chunks):
                    await conn.execute("""
                        INSERT INTO chunks (document_id, chunk_index, content, content_type, page_number, position, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (document_id, chunk_index) DO UPDATE
                        SET content = EXCLUDED.content, content_type = EXCLUDED.content_type
                    """, doc["id"], idx, chunk["content"], chunk["type"], chunk.get("page"), chunk.get("position"), json.dumps(chunk.get("metadata", {})))

                await conn.execute("""
                    UPDATE documents 
                    SET status = 'completed', processed_at = NOW()
                    WHERE id = $1
                """, doc["id"])

                await conn.execute("""
                    UPDATE processing_queue 
                    SET status = 'completed', completed_at = NOW()
                    WHERE id = $1
                """, task["id"])

                await conn.execute("""
                    INSERT INTO processing_queue (document_id, task_type, priority)
                    VALUES ($1, 'enhance', $2)
                """, doc["id"], task["priority"])

                logger.info(f"Processed document {doc['original_name']}: {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Document processing error: {e}")
                await conn.execute("UPDATE processing_queue SET status = 'failed', error_message = $2, completed_at = NOW() WHERE id = $1", task["id"], str(e))
                await conn.execute("UPDATE documents SET status = 'failed' WHERE id = $1", task["document_id"])

    async def extract_content(self, doc):
        file_type = doc['file_type'].lower()
        metadata = doc.get('metadata', {})

        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        logger.info(f"Metadata keys: {list(metadata.keys())}")

        file_path = metadata.get('file_path')
        file_content = metadata.get('file_content') or metadata.get('content')

        logger.info(f"file_content (raw preview): {str(file_content)[:60]}")

        if file_content:
            try:
                file_bytes = base64.b64decode(file_content.encode("utf-8"))
                logger.info("✅ Successfully decoded base64 content")
            except Exception as e:
                logger.error(f"Base64 decode error: {e}")
                raise ValueError("Invalid base64 content")
        elif file_path and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
        else:
            raise ValueError("No file content available")

        if file_type == "pdf":
            return await self.extract_pdf(file_bytes)
        elif file_type in ["png", "jpg", "jpeg", "tiff", "tif"]:
            return await self.extract_image(file_bytes)
        elif file_type in ["txt", "md", "log"]:
            return self.extract_text(file_bytes.decode("utf-8", errors="ignore"))
        else:
            return [{"content": f"[Unsupported file type: {file_type}]", "type": "text"}]

    async def extract_pdf(self, pdf_bytes):
        chunks = []
        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=300)
        for page_num, image in enumerate(images, 1):
            text = pytesseract.image_to_string(image, lang=os.getenv("TESSERACT_LANG", "deu+eng"))
            if text.strip():
                for chunk in self.split_text(text):
                    chunks.append({"content": chunk, "type": "text", "page": page_num})
        return chunks

    async def extract_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang=os.getenv("TESSERACT_LANG", "deu+eng"))
        if text.strip():
            return [{"content": chunk, "type": "text", "page": 1} for chunk in self.split_text(text)]
        return [{"content": "[Bild ohne erkannten Text]", "type": "text", "page": 1}]

    def extract_text(self, text):
        return [{"content": chunk, "type": "text"} for chunk in self.split_text(text)]

    def split_text(self, text):
        words = text.split()
        chunks = []
        current = []
        current_size = 0
        for word in words:
            current.append(word)
            current_size += len(word) + 1
            if current_size >= self.chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_size = 0
        if current:
            chunks.append(" ".join(current))
        return chunks or [""]

    async def run(self):
        await self.connect()
        logger.info("OCR Agent V2 started")
        try:
            await self.process_queue()
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            await self.close()

def main():
    db_url = os.getenv("DATABASE_URL", "postgresql://semanticuser:semantic2024@postgres:5432/semanticuser")
    agent = OCRAgentV2(db_url)
    asyncio.run(agent.run())

if __name__ == "__main__":
    main()
