# ocr_agent_optimized.py
# Optimierter OCR Agent mit besten Features aus beiden Versionen

import asyncio
import base64
import os
import gc
import json
import logging
import re
import hashlib
import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal

# Externe Abhängigkeiten
import asyncpg
import pytesseract
from PIL import Image
import pdf2image
from pydantic import BaseModel, Field, validator

# Bedingte Imports
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

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logging.warning("beautifulsoup4 not available - HTML parsing limited")

# Email parsing
import email
from email.parser import BytesParser
from email import policy

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OCRAgentOptimized")


# === PYDANTIC MODELS ===

class DocumentMetadata(BaseModel):
    """Strukturierte Metadaten mit N8N-Kompatibilität"""
    
    # Content sources
    file_content: Optional[str] = None
    file_path: Optional[str] = None
    
    # Processing configuration
    processing_mode: Literal["auto", "streaming", "in_memory"] = "auto"
    source: str = "unknown"
    
    # File metadata
    original_filename: Optional[str] = None
    file_size_mb: Optional[Union[float, str]] = None
    
    # N8N compatibility fields (camelCase)
    fileContent: Optional[str] = None
    filePath: Optional[str] = None
    
    # Additional N8N fields
    fileName: Optional[str] = None
    fileType: Optional[str] = None
    
    # Additional metadata
    uploadedAt: Optional[str] = None
    processed: bool = False
    
    class Config:
        extra = "allow"  # Allow additional fields
    
    @validator('file_size_mb', pre=True)
    def convert_file_size(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        return None
    
    def get_file_content(self) -> Optional[str]:
        """Get file content with fallback"""
        return self.file_content or self.fileContent
    
    def get_file_path(self) -> Optional[str]:
        """Get file path with validation"""
        path = self.file_path or self.filePath
        if path and os.path.exists(path) and os.path.isfile(path):
            return path
        return None
    
    def get_original_filename(self) -> Optional[str]:
        """Get filename with fallback"""
        return self.original_filename or self.fileName
    
    def determine_processing_mode(self) -> str:
        """Auto-determine processing mode based on file size"""
        if self.processing_mode != "auto":
            return self.processing_mode
            
        # Check file size
        if self.get_file_path():
            try:
                file_size = os.path.getsize(self.get_file_path())
                file_size_mb = file_size / (1024 * 1024)
                
                # Auto-decide based on size
                if file_size_mb > 100:
                    return "streaming"
                else:
                    return "in_memory"
            except:
                pass
        
        return "in_memory"  # Default


class OCRAgentOptimized:
    """
    Optimierter OCR Agent mit N8N-Integration und robuster Fehlerbehandlung
    """
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.db_pool = None
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.running = True
        
        # Size limits
        self.MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
        self.MAX_BASE64_LENGTH = 700 * 1024 * 1024  # 700MB string
        
        # Tesseract configs
        self.tesseract_config = "--oem 3 --psm 6"
        self.tesseract_config_fast = "--oem 1 --psm 6"
        
        logger.info(f"📊 OCR Agent Optimized initialisiert:")
        logger.info(f"   • Chunk-Größe: {self.chunk_size}")
        logger.info(f"   • Max File Size: {self.MAX_FILE_SIZE / (1024*1024):.0f}MB")
        logger.info(f"   • DB URL: {db_url.split('@')[1] if '@' in db_url else 'localhost'}")

    async def connect(self):
        """Verbinde mit der PostgreSQL-Datenbank"""
        try:
            self.db_pool = await asyncpg.create_pool(self.db_url)
            logger.info("✅ Verbindung zur PostgreSQL-Datenbank hergestellt")
        except Exception as e:
            logger.error(f"❌ Datenbankverbindung fehlgeschlagen: {e}")
            raise

    async def close(self):
        """Schließe die Verbindung zur Datenbank"""
        if self.db_pool:
            await self.db_pool.close()
        logger.info("🛑 OCR Agent beendet")

    async def process_queue(self):
        """Verarbeite die Warteschlange kontinuierlich"""
        while self.running:
            try:
                async with self.db_pool.acquire() as conn:
                    task = await conn.fetchrow(
                        """
                        UPDATE processing_queue 
                        SET status = 'processing', started_at = NOW()
                        WHERE id = (
                            SELECT id FROM processing_queue 
                            WHERE status = 'pending' 
                            AND task_type IN ('ocr', 'extract')
                            AND (NOT deferred OR EXTRACT(hour FROM NOW()) BETWEEN 22 AND 6)
                            ORDER BY priority DESC, created_at ASC
                            LIMIT 1
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING *
                        """
                    )
                    if task:
                        await self.process_document(task)
                    else:
                        await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"⚠️ Fehler bei Warteschlange: {e}")
                await asyncio.sleep(10)

    async def process_document(self, task: Dict):
        """Verarbeite ein Dokument"""
        async with self.db_pool.acquire() as conn:
            try:
                doc = await conn.fetchrow(
                    "SELECT * FROM documents WHERE id = $1",
                    task["document_id"]
                )
                if not doc:
                    raise ValueError(f"🚫 Dokument {task['document_id']} nicht gefunden")

                logger.info(f"📄 Verarbeite: {doc['original_name']} (ID: {doc['id']})")

                # Extrahiere Inhalt
                chunks = await self.extract_content(doc)
                
                if not chunks:
                    logger.warning(f"⚠️ Keine Chunks erstellt für {doc['original_name']}")
                    chunks = [self._create_error_chunk("Keine Inhalte extrahiert")]

                # Validiere und speichere Chunks
                chunks = self._validate_chunks(chunks)
                await self._save_chunks_to_db(conn, doc["id"], chunks)

                # Update document status
                await conn.execute(
                    """UPDATE documents 
                       SET status = 'completed', processed_at = NOW()
                       WHERE id = $1""",
                    doc["id"]
                )

                # Complete task
                await conn.execute(
                    """UPDATE processing_queue 
                       SET status = 'completed', completed_at = NOW()
                       WHERE id = $1""",
                    task["id"]
                )

                # Create enhancement task
                await conn.execute(
                    """INSERT INTO processing_queue (document_id, task_type, priority)
                       VALUES ($1, 'enhance', $2)""",
                    doc["id"], task["priority"]
                )

                logger.info(f"✅ Dokument verarbeitet: {doc['original_name']} ({len(chunks)} Chunks)")

            except Exception as e:
                logger.error(f"❌ Fehler bei Dokumentenverarbeitung: {e}")
                await self._mark_task_failed(conn, task["id"], task["document_id"], str(e))

    async def extract_content(self, doc: Dict) -> List[Dict]:
        """Hauptmethode für Content-Extraktion"""
        try:
            file_type = doc["file_type"].lower()
            
            # Map MIME types to simple extensions
            file_type_mapping = {
                "application/pdf": "pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                "application/msword": "doc",
                "text/plain": "txt",
                "text/html": "html",
                "image/png": "png",
                "image/jpeg": "jpg",
            }
            file_type = file_type_mapping.get(file_type, file_type.split('/')[-1])
            
            metadata = self._parse_metadata_robust(doc.get("metadata", {}))
            
            # Log metadata for debugging
            logger.debug(f"📋 Parsed metadata: file_content={bool(metadata.get_file_content())}, "
                         f"file_path={metadata.get_file_path()}, "
                         f"original_filename={metadata.get_original_filename()}")
            
            # Determine processing mode
            processing_mode = metadata.determine_processing_mode()
            logger.info(f"🔍 Processing mode: {processing_mode}, file_type: {file_type}")
            
            # Get file source
            file_path = metadata.get_file_path()
            file_content = metadata.get_file_content()
            
            # Streaming mode
            if processing_mode == "streaming" and file_path:
                return await self._extract_content_streaming(file_path, file_type, doc["original_name"])
            
            # In-memory mode
            file_bytes = await self._get_file_bytes(file_content, file_path)
            if not file_bytes:
                raise ValueError("Keine Dateidaten gefunden")
                
            return await self._dispatch_extraction(file_type, file_bytes, doc["original_name"])
            
        except Exception as e:
            logger.error(f"❌ Content-Extraktion fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"Content-Extraktion fehlgeschlagen: {e}")]

    def _parse_metadata_robust(self, raw_metadata: Union[str, dict]) -> DocumentMetadata:
        """Parse metadata to DocumentMetadata object"""
        try:
            # Convert string to dict if necessary
            if isinstance(raw_metadata, str):
                try:
                    metadata_dict = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ JSON-Parsing fehlgeschlagen")
                    metadata_dict = {}
            elif isinstance(raw_metadata, dict):
                metadata_dict = raw_metadata
            else:
                metadata_dict = {}
            
            # Log what we got
            logger.debug(f"📋 Metadata keys: {list(metadata_dict.keys())}")
            
            # Create DocumentMetadata with error handling
            try:
                return DocumentMetadata(**metadata_dict)
            except Exception as e:
                logger.warning(f"⚠️ DocumentMetadata validation failed: {e}")
                # Return minimal valid metadata
                return DocumentMetadata(
                    file_content=metadata_dict.get('file_content') or metadata_dict.get('fileContent'),
                    file_path=metadata_dict.get('file_path') or metadata_dict.get('filePath'),
                    original_filename=metadata_dict.get('original_filename') or metadata_dict.get('fileName'),
                    source=metadata_dict.get('source', 'unknown')
                )
                
        except Exception as e:
            logger.error(f"❌ Metadata-Parsing komplett fehlgeschlagen: {e}")
            return DocumentMetadata()

    async def _get_file_bytes(self, file_content: Optional[str], file_path: Optional[str]) -> Optional[bytes]:
        """Get file bytes from content or path"""
        logger.debug(f"🔍 Suche Dateidaten - file_content: {bool(file_content)}, file_path: {file_path}")
        
        # Try base64 content first
        if file_content:
            if self._validate_base64(file_content):
                try:
                    cleaned = self._clean_base64(file_content)
                    return base64.b64decode(cleaned)
                except Exception as e:
                    logger.warning(f"⚠️ Base64-Dekodierung fehlgeschlagen: {e}")
        
        # Try file path
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"⚠️ Datei-Lesen fehlgeschlagen: {e}")
        
        logger.error(f"❌ Keine Dateidaten gefunden - content: {file_content[:50] if file_content else 'None'}, path: {file_path}")
        return None

    def _validate_base64(self, content: str) -> bool:
        """Validate base64 content"""
        if not content or not isinstance(content, str):
            return False
            
        # Remove data URL prefix if present
        if content.startswith("data:"):
            content = content.split(",", 1)[-1] if "," in content else content
            
        # Clean and check pattern
        cleaned = content.strip().replace('\n', '').replace(' ', '')
        
        # Check base64 pattern
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', cleaned):
            return False
            
        # Try test decode
        try:
            test = cleaned[:1000] if len(cleaned) > 1000 else cleaned
            base64.b64decode(test + '=' * (4 - len(test) % 4))
            return True
        except:
            return False

    def _clean_base64(self, content: str) -> str:
        """Clean base64 content"""
        # Remove data URL prefix
        if content.startswith("data:") and "," in content:
            content = content.split(",", 1)[1]
        
        # Remove whitespace
        return content.strip().replace('\n', '').replace(' ', '').replace('\r', '')

    async def _dispatch_extraction(self, file_type: str, file_bytes: bytes, filename: str) -> List[Dict]:
        """Dispatch to specific extraction method"""
        try:
            if file_type == "pdf":
                return await self.extract_pdf(file_bytes, filename)
            elif file_type in ["png", "jpg", "jpeg", "tiff", "tif"]:
                return await self.extract_image(file_bytes)
            elif file_type in ["txt", "md", "log"]:
                content = file_bytes.decode("utf-8", errors="ignore")
                return self.extract_text(content, file_type)
            elif file_type in ["docx", "doc"]:
                return await self.extract_docx(file_bytes)
            elif file_type in ["xlsx", "xls", "csv"]:
                return await self.extract_spreadsheet(file_bytes, file_type)
            elif file_type == "eml":
                return await self.extract_email(file_bytes)
            elif file_type == "html":
                content = file_bytes.decode("utf-8", errors="ignore")
                return self.extract_html(content)
            elif file_type == "zip":
                return await self.extract_archive(file_bytes)
            else:
                # Default to text
                content = file_bytes.decode("utf-8", errors="ignore")
                return self.extract_text(content, file_type)
                
        except Exception as e:
            logger.error(f"❌ Extraktion für {file_type} fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"Extraktion fehlgeschlagen: {e}")]

    # === EXTRACTION METHODS ===

    async def extract_pdf(self, pdf_bytes: bytes, filename: str) -> List[Dict]:
        """Extract text from PDF"""
        chunks = []
        try:
            images = pdf2image.convert_from_bytes(pdf_bytes, dpi=300)
            logger.info(f"📄 PDF: {filename} ({len(images)} Seiten)")
            
            for page_num, image in enumerate(images, 1):
                text = pytesseract.image_to_string(
                    image,
                    lang=os.getenv("TESSERACT_LANG", "deu+eng"),
                    config=self.tesseract_config
                )
                
                if text.strip():
                    page_chunks = self.split_text(text)
                    for chunk_text in page_chunks:
                        chunks.append({
                            "content": chunk_text,
                            "type": "text",
                            "page": page_num,
                            "metadata": {
                                "extraction_method": "pdf_ocr",
                                "page_total": len(images)
                            }
                        })
                
                # Memory cleanup
                del image
                if page_num % 5 == 0:
                    gc.collect()
                    
        except Exception as e:
            logger.error(f"❌ PDF-Extraktion fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"PDF-Fehler: {e}")]
            
        return chunks or [self._create_error_chunk("Kein Text aus PDF extrahiert")]

    async def extract_image(self, image_bytes: bytes) -> List[Dict]:
        """Extract text from image"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(
                image,
                lang=os.getenv("TESSERACT_LANG", "deu+eng"),
                config=self.tesseract_config
            )
            
            if not text.strip():
                return [{"content": "[Bild ohne Text]", "type": "image", "metadata": {}}]
                
            chunks = self.split_text(text)
            return [{
                "content": chunk,
                "type": "text",
                "page": 1,
                "metadata": {"extraction_method": "image_ocr"}
            } for chunk in chunks]
            
        except Exception as e:
            logger.error(f"❌ Bild-Extraktion fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"Bild-Fehler: {e}")]

    def extract_text(self, text: str, file_type: str = "txt") -> List[Dict]:
        """Extract text with chunking"""
        if not text.strip():
            return [{"content": "[Leerer Text]", "type": "text", "metadata": {}}]
            
        chunks = self.split_text(text)
        return [{
            "content": chunk,
            "type": "text",
            "metadata": {
                "extraction_method": "text_direct",
                "file_type": file_type
            }
        } for chunk in chunks]

    async def extract_docx(self, docx_bytes: bytes) -> List[Dict]:
        """Extract text from DOCX"""
        if not HAS_DOCX:
            return [self._create_error_chunk("DOCX-Support nicht verfügbar")]
            
        try:
            doc = DocxDocument(io.BytesIO(docx_bytes))
            text = "\n".join([para.text for para in doc.paragraphs])
            
            result = self.extract_text(text, "docx")
            
            # Extract tables
            for idx, table in enumerate(doc.tables):
                table_data = []
                for row in table.rows:
                    table_data.append([cell.text for cell in row.cells])
                    
                if table_data:
                    result.append({
                        "content": json.dumps(table_data),
                        "type": "table",
                        "metadata": {
                            "extraction_method": "docx_table",
                            "table_index": idx
                        }
                    })
                    
            return result
            
        except Exception as e:
            logger.error(f"❌ DOCX-Extraktion fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"DOCX-Fehler: {e}")]

    async def extract_spreadsheet(self, file_bytes: bytes, file_type: str) -> List[Dict]:
        """Extract data from spreadsheet"""
        if not HAS_PANDAS:
            return [self._create_error_chunk("Spreadsheet-Support nicht verfügbar")]
            
        try:
            if file_type == "csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
            else:
                df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
                
            chunks = []
            
            if isinstance(df, dict):  # Multiple sheets
                for sheet_name, sheet_df in df.items():
                    chunks.extend(self._process_dataframe(sheet_df, {"sheet": sheet_name}))
            else:
                chunks.extend(self._process_dataframe(df))
                
            return chunks or [self._create_error_chunk("Keine Daten extrahiert")]
            
        except Exception as e:
            logger.error(f"❌ Spreadsheet-Extraktion fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"Spreadsheet-Fehler: {e}")]

    def _process_dataframe(self, df: "pd.DataFrame", metadata: dict = None) -> List[Dict]:
        """Process DataFrame into chunks"""
        chunks = []
        metadata = metadata or {}
        
        try:
            # Split into chunks of 50 rows
            for i in range(0, len(df), 50):
                chunk_df = df.iloc[i:i + 50]
                chunks.append({
                    "content": chunk_df.to_json(orient="records"),
                    "type": "table",
                    "metadata": {
                        **metadata,
                        "row_start": i,
                        "row_count": len(chunk_df),
                        "extraction_method": "pandas"
                    }
                })
        except Exception as e:
            logger.error(f"DataFrame-Verarbeitung fehlgeschlagen: {e}")
            
        return chunks

    async def extract_email(self, eml_bytes: bytes) -> List[Dict]:
        """Extract email content"""
        try:
            msg = BytesParser(policy=policy.default).parsebytes(eml_bytes)
            
            # Email metadata
            chunks = [{
                "content": json.dumps({
                    "from": msg["From"],
                    "to": msg["To"],
                    "subject": msg["Subject"],
                    "date": msg["Date"],
                }),
                "type": "metadata",
                "metadata": {"extraction_method": "email_header"}
            }]
            
            # Email body
            body = msg.get_body(preferencelist=("plain", "html"))
            if body:
                content = body.get_content()
                if content and content.strip():
                    text_chunks = self.split_text(content)
                    chunks.extend([{
                        "content": chunk,
                        "type": "text",
                        "metadata": {"extraction_method": "email_body"}
                    } for chunk in text_chunks])
                    
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Email-Extraktion fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"Email-Fehler: {e}")]

    def extract_html(self, html: str) -> List[Dict]:
        """Extract text from HTML"""
        try:
            if HAS_BS4:
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
            else:
                # Simple regex fallback
                text = re.sub('<[^<]+?>', '', html)
                
            return self.extract_text(text, "html")
            
        except Exception as e:
            logger.error(f"❌ HTML-Extraktion fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"HTML-Fehler: {e}")]

    async def extract_archive(self, zip_bytes: bytes) -> List[Dict]:
        """Extract file list from archive"""
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                files = zf.namelist()
                return [{
                    "content": f"Archive enthält: {name}",
                    "type": "metadata",
                    "metadata": {
                        "archived_file": name,
                        "extraction_method": "zip_listing"
                    }
                } for name in files if not name.endswith("/")]
                
        except Exception as e:
            logger.error(f"❌ Archiv-Extraktion fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"Archiv-Fehler: {e}")]

    # === STREAMING METHODS ===

    async def _extract_content_streaming(self, file_path: str, file_type: str, filename: str) -> List[Dict]:
        """Streaming extraction for large files"""
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"🚀 Streaming für {filename}: {file_size_mb:.1f}MB")
            
            if file_type == "pdf":
                return await self._stream_large_pdf(file_path)
            elif file_type in ["txt", "md", "log"]:
                return await self._stream_large_text(file_path)
            elif file_type in ["docx", "doc"] and HAS_DOCX:
                return await self._stream_large_docx(file_path)
            else:
                # Fallback to normal loading
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                return await self._dispatch_extraction(file_type, file_bytes, filename)
                
        except Exception as e:
            logger.error(f"❌ Streaming fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"Streaming-Fehler: {e}")]

    async def _stream_large_pdf(self, file_path: str) -> List[Dict]:
        """Stream large PDF page by page"""
        chunks = []
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            dpi = 150 if file_size_mb > 50 else 300
            config = self.tesseract_config_fast if file_size_mb > 100 else self.tesseract_config
            
            images = pdf2image.convert_from_path(file_path, dpi=dpi)
            total_pages = len(images)
            
            for page_num, image in enumerate(images, 1):
                if page_num % 10 == 0:
                    logger.info(f"📄 Progress: {page_num}/{total_pages}")
                
                text = pytesseract.image_to_string(image, lang="deu+eng", config=config)
                
                if text.strip():
                    page_chunks = self.split_text(text)
                    for chunk in page_chunks:
                        chunks.append({
                            "content": chunk,
                            "type": "text",
                            "page": page_num,
                            "metadata": {
                                "extraction_method": "pdf_streaming",
                                "dpi": dpi
                            }
                        })
                
                del image
                if page_num % 5 == 0:
                    gc.collect()
                    
        except Exception as e:
            logger.error(f"❌ PDF-Streaming fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"PDF-Streaming-Fehler: {e}")]
            
        return chunks

    async def _stream_large_text(self, file_path: str) -> List[Dict]:
        """Stream large text file"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    current_chunk.append(line)
                    current_size += len(line)
                    
                    if current_size >= self.chunk_size:
                        chunks.append({
                            "content": "".join(current_chunk),
                            "type": "text",
                            "metadata": {"extraction_method": "text_streaming"}
                        })
                        current_chunk = []
                        current_size = 0
                        
                # Last chunk
                if current_chunk:
                    chunks.append({
                        "content": "".join(current_chunk),
                        "type": "text",
                        "metadata": {"extraction_method": "text_streaming"}
                    })
                    
        except Exception as e:
            logger.error(f"❌ Text-Streaming fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"Text-Streaming-Fehler: {e}")]
            
        return chunks

    async def _stream_large_docx(self, file_path: str) -> List[Dict]:
        """Stream large DOCX file"""
        if not HAS_DOCX:
            return [self._create_error_chunk("DOCX-Support nicht verfügbar")]
            
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            chunks = self.split_text(text)
            
            result = [{
                "content": chunk,
                "type": "text",
                "metadata": {"extraction_method": "docx_streaming"}
            } for chunk in chunks]
            
            # Extract tables
            for idx, table in enumerate(doc.tables):
                table_data = []
                for row in table.rows:
                    table_data.append([cell.text for cell in row.cells])
                if table_data:
                    result.append({
                        "content": json.dumps(table_data),
                        "type": "table",
                        "metadata": {"table_index": idx}
                    })
                    
            return result
            
        except Exception as e:
            logger.error(f"❌ DOCX-Streaming fehlgeschlagen: {e}")
            return [self._create_error_chunk(f"DOCX-Streaming-Fehler: {e}")]

    # === UTILITY METHODS ===

    def split_text(self, text: str, chunk_size: int = None) -> List[str]:
        """Split text into chunks"""
        chunk_size = chunk_size or self.chunk_size
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks or [""]

    def _create_error_chunk(self, error_message: str) -> Dict:
        """Create error chunk"""
        return {
            "content": f"[FEHLER: {error_message}]",
            "type": "error",
            "metadata": {
                "extraction_method": "error",
                "error_message": error_message,
                "timestamp": datetime.now().isoformat()
            }
        }

    def _validate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Validate chunk format"""
        validated = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                chunk = {"content": str(chunk), "type": "text", "metadata": {}}
                
            # Ensure required fields
            chunk.setdefault("content", "[Leer]")
            chunk.setdefault("type", "text")
            chunk.setdefault("metadata", {})
            
            validated.append(chunk)
            
        return validated

    async def _save_chunks_to_db(self, conn, document_id: str, chunks: List[Dict]):
        """Save chunks to database"""
        for idx, chunk in enumerate(chunks):
            await conn.execute(
                """
                INSERT INTO chunks (
                    document_id, chunk_index, content, content_type,
                    page_number, position, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (document_id, chunk_index) DO UPDATE
                SET content = EXCLUDED.content,
                    content_type = EXCLUDED.content_type,
                    metadata = EXCLUDED.metadata
                """,
                document_id,
                idx,
                chunk["content"],
                chunk.get("type", "text"),
                chunk.get("page"),
                chunk.get("position"),
                json.dumps(chunk.get("metadata", {}))
            )

    async def _mark_task_failed(self, conn, task_id: str, doc_id: str, error_message: str):
        """Mark task as failed"""
        await conn.execute(
            """UPDATE processing_queue 
               SET status = 'failed', error_message = $2, completed_at = NOW()
               WHERE id = $1""",
            task_id, error_message
        )
        await conn.execute(
            """UPDATE documents SET status = 'failed' WHERE id = $1""",
            doc_id
        )

    async def run(self):
        """Main run loop"""
        await self.connect()
        logger.info("🧠 OCR Agent Optimized gestartet")
        
        try:
            await self.process_queue()
        except KeyboardInterrupt:
            logger.info("🛑 OCR Agent wird beendet")
        except Exception as e:
            logger.error(f"💥 Unerwarteter Fehler: {e}")
            raise
        finally:
            await self.close()


def main():
    """Main entry point"""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://semanticuser:semantic2024@postgres:5432/semantic_doc_finder"
    )
    
    agent = OCRAgentOptimized(db_url)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()