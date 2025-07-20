import asyncio
import base64
import json
import logging
import os
import io
import asyncpg
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pdf2image
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2

# Optional imports with fallbacks
try:
    import pandas as pd
    from tabulate import tabulate
    HAS_TABULATION = True
except ImportError:
    HAS_TABULATION = False
    logging.warning("pandas/tabulate not available - enhanced table processing disabled")

try:
    import camelot  # For table extraction from PDFs
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False
    logging.warning("camelot not available - PDF table extraction limited")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enhanced_ocr_agent")

class EnhancedOCRAgent:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.db_pool = None
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.running = True
        
        # OCR Configuration
        self.tesseract_config = '--oem 3 --psm 6'  # Best for general text
        self.table_config = '--oem 3 --psm 6'     # Good for tabular data
        self.image_config = '--oem 3 --psm 8'     # Single word mode for images

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

                chunks = await self.extract_content_enhanced(doc)

                # Save all chunks
                for idx, chunk in enumerate(chunks):
                    await conn.execute("""
                        INSERT INTO chunks (document_id, chunk_index, content, content_type, page_number, position, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (document_id, chunk_index) DO UPDATE
                        SET content = EXCLUDED.content, content_type = EXCLUDED.content_type, metadata = EXCLUDED.metadata
                    """, doc["id"], idx, chunk["content"], chunk["type"], 
                        chunk.get("page"), chunk.get("position"), json.dumps(chunk.get("metadata", {})))

                # Create table_content entries for tables
                for chunk in chunks:
                    if chunk["type"] == "table" and "table_data" in chunk.get("metadata", {}):
                        # We'll handle this later when enhanced_chunks table is populated
                        pass

                # Create image_content entries for images  
                for chunk in chunks:
                    if chunk["type"] == "image" and "image_base64" in chunk.get("metadata", {}):
                        # We'll handle this later when enhanced_chunks table is populated
                        pass

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

    async def extract_content_enhanced(self, doc: Dict) -> List[Dict]:
        """Enhanced content extraction with table and image support"""
        file_type = doc['file_type'].lower()
        metadata = doc.get('metadata', {})

        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        file_content = metadata.get('file_content')
        file_path = metadata.get('file_path')

        if file_content:
            try:
                file_bytes = base64.b64decode(file_content.encode("utf-8"))
            except Exception as e:
                logger.error(f"Base64 decode error: {e}")
                raise ValueError("Invalid base64 content")
        elif file_path and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
        else:
            raise ValueError("No file content available")

        if file_type == "pdf":
            return await self.extract_pdf_enhanced(file_bytes, doc['original_name'])
        elif file_type in ["png", "jpg", "jpeg", "tiff", "tif"]:
            return await self.extract_image_enhanced(file_bytes)
        elif file_type in ["txt", "md", "log"]:
            return self.extract_text(file_bytes.decode("utf-8", errors="ignore"))
        else:
            return [{"content": f"[Unsupported file type: {file_type}]", "type": "text"}]

    async def extract_pdf_enhanced(self, pdf_bytes: bytes, filename: str) -> List[Dict]:
        """Enhanced PDF extraction with table detection and image extraction"""
        chunks = []
        
        # Convert PDF to images
        try:
            images = pdf2image.convert_from_bytes(pdf_bytes, dpi=300)
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return [{"content": f"[PDF conversion failed: {e}]", "type": "error"}]

        # Try table extraction with camelot first (if available)
        if HAS_CAMELOT:
            try:
                # Save PDF temporarily for camelot
                temp_pdf_path = f"/tmp/temp_pdf_{hash(pdf_bytes)}.pdf"
                with open(temp_pdf_path, 'wb') as f:
                    f.write(pdf_bytes)
                
                tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='lattice')
                logger.info(f"Camelot found {len(tables)} tables in {filename}")
                
                for i, table in enumerate(tables):
                    if table.parsing_report['accuracy'] > 50:  # Only use high-confidence tables
                        table_data = table.df.to_dict('records')
                        chunks.append({
                            "content": f"Table {i+1} from page {table.page} (accuracy: {table.parsing_report['accuracy']:.1f}%)",
                            "type": "table",
                            "page": table.page,
                            "metadata": {
                                "table_data": table_data,
                                "table_headers": table.df.columns.tolist(),
                                "accuracy": table.parsing_report['accuracy'],
                                "table_index": i,
                                "extraction_method": "camelot"
                            }
                        })
                
                os.unlink(temp_pdf_path)  # Cleanup
            except Exception as e:
                logger.warning(f"Camelot table extraction failed: {e}")

        # Process each page
        for page_num, image in enumerate(images, 1):
            logger.info(f"Processing page {page_num} of {filename}")
            
            # 1. Text extraction with OCR
            text_chunks = await self.extract_text_from_image(image, page_num)
            chunks.extend(text_chunks)
            
            # 2. Table detection and extraction
            table_chunks = await self.detect_and_extract_tables(image, page_num)
            chunks.extend(table_chunks)
            
            # 3. Image/diagram extraction
            image_chunks = await self.extract_images_from_page(image, page_num)
            chunks.extend(image_chunks)

        return chunks

    async def extract_text_from_image(self, image: Image.Image, page_num: int) -> List[Dict]:
        """Extract text from image with preprocessing"""
        chunks = []
        
        # Preprocess image for better OCR
        processed_image = self.preprocess_image_for_ocr(image)
        
        # Extract text
        text = pytesseract.image_to_string(processed_image, 
                                         lang=os.getenv("TESSERACT_LANG", "deu+eng"),
                                         config=self.tesseract_config)
        
        if text.strip():
            text_chunks = self.split_text(text)
            for chunk_text in text_chunks:
                chunks.append({
                    "content": chunk_text,
                    "type": "text",
                    "page": page_num,
                    "metadata": {
                        "extraction_method": "tesseract_enhanced",
                        "confidence": "medium"  # Could be enhanced with actual confidence scores
                    }
                })
        
        return chunks

    async def detect_and_extract_tables(self, image: Image.Image, page_num: int) -> List[Dict]:
        """Detect and extract tables from image"""
        if not HAS_TABULATION:
            return []
            
        chunks = []
        
        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Table detection using line detection
        tables_detected = self.detect_table_regions(gray)
        
        for i, table_region in enumerate(tables_detected):
            # Extract table region
            x, y, w, h = table_region
            table_img = image.crop((x, y, x+w, y+h))
            
            # OCR on table with special config
            table_text = pytesseract.image_to_string(table_img, 
                                                   lang=os.getenv("TESSERACT_LANG", "deu+eng"),
                                                   config=self.table_config)
            
            if table_text.strip():
                # Try to parse as structured table
                table_data = self.parse_table_text(table_text)
                
                chunks.append({
                    "content": f"Table {i+1} detected on page {page_num}:\n{table_text}",
                    "type": "table",
                    "page": page_num,
                    "position": {"x": x, "y": y, "width": w, "height": h},
                    "metadata": {
                        "table_data": table_data,
                        "original_image_base64": self.image_to_base64(table_img),
                        "extraction_method": "opencv_detection",
                        "table_index": i,
                        "ocr_confidence": "medium"
                    }
                })
        
        return chunks

    async def extract_images_from_page(self, image: Image.Image, page_num: int) -> List[Dict]:
        """Extract images, diagrams, and charts from page"""
        chunks = []
        
        # Convert to OpenCV
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect non-text regions (potential images/diagrams)
        image_regions = self.detect_image_regions(cv_image)
        
        for i, region in enumerate(image_regions):
            x, y, w, h = region
            
            # Skip very small regions
            if w < 100 or h < 100:
                continue
                
            extracted_img = image.crop((x, y, x+w, y+h))
            
            # OCR on the image (might contain text within diagrams)
            img_text = pytesseract.image_to_string(extracted_img,
                                                 lang=os.getenv("TESSERACT_LANG", "deu+eng"),
                                                 config=self.image_config)
            
            # Classify image type (basic heuristics)
            img_type = self.classify_image_content(extracted_img, img_text)
            
            content_description = f"Image/Diagram {i+1} on page {page_num}"
            if img_text.strip():
                content_description += f" - Contains text: {img_text[:100]}..."
            
            chunks.append({
                "content": content_description,
                "type": "image",
                "page": page_num,
                "position": {"x": x, "y": y, "width": w, "height": h},
                "metadata": {
                    "image_base64": self.image_to_base64(extracted_img),
                    "image_format": "png",
                    "image_dimensions": {"width": w, "height": h},
                    "ocr_text": img_text.strip(),
                    "image_type": img_type,
                    "extraction_method": "opencv_region_detection"
                }
            })
        
        return chunks

    def preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)
        
        # Resize if too small (OCR works better on larger images)
        width, height = image.size
        if width < 1000:
            scale_factor = 1000 / width
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image

    def detect_table_regions(self, gray_image) -> List[Tuple[int, int, int, int]]:
        """Detect table regions using line detection"""
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Apply morphological operations
        horizontal_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out small regions
            if w > 200 and h > 100:
                tables.append((x, y, w, h))
        
        return tables

    def detect_image_regions(self, cv_image) -> List[Tuple[int, int, int, int]]:
        """Detect non-text regions that might contain images or diagrams"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection to find complex regions
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter for reasonable image sizes
            if w > 150 and h > 150 and w < cv_image.shape[1] * 0.8 and h < cv_image.shape[0] * 0.8:
                regions.append((x, y, w, h))
        
        return regions

    def parse_table_text(self, table_text: str) -> List[Dict]:
        """Parse OCR text into structured table data"""
        if not HAS_TABULATION:
            return []
            
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return []
        
        # Simple table parsing - assumes first line is header
        try:
            header = [col.strip() for col in lines[0].split() if col.strip()]
            rows = []
            
            for line in lines[1:]:
                row_data = [col.strip() for col in line.split() if col.strip()]
                if len(row_data) == len(header):
                    rows.append(dict(zip(header, row_data)))
            
            return rows
        except Exception as e:
            logger.warning(f"Table parsing failed: {e}")
            return []

    def classify_image_content(self, image: Image.Image, ocr_text: str) -> str:
        """Basic image content classification"""
        # Simple heuristics for image classification
        if any(word in ocr_text.lower() for word in ['chart', 'graph', 'diagram', '%', 'axis']):
            return 'chart'
        elif any(word in ocr_text.lower() for word in ['logo', 'company', 'brand']):
            return 'logo'
        elif len(ocr_text.strip()) > 50:
            return 'text_image'
        else:
            return 'diagram'

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    async def extract_image_enhanced(self, image_bytes: bytes) -> List[Dict]:
        """Enhanced single image processing"""
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess for better OCR
        processed_image = self.preprocess_image_for_ocr(image)
        
        # Extract text
        text = pytesseract.image_to_string(processed_image, 
                                         lang=os.getenv("TESSERACT_LANG", "deu+eng"))
        
        chunks = []
        if text.strip():
            text_chunks = self.split_text(text)
            for chunk_text in text_chunks:
                chunks.append({
                    "content": chunk_text,
                    "type": "text",
                    "page": 1,
                    "metadata": {
                        "extraction_method": "single_image_ocr",
                        "original_image_base64": base64.b64encode(image_bytes).decode()
                    }
                })
        else:
            chunks.append({
                "content": "[Image without readable text]",
                "type": "image",
                "page": 1,
                "metadata": {
                    "image_base64": base64.b64encode(image_bytes).decode(),
                    "image_format": image.format or "unknown",
                    "image_dimensions": {"width": image.width, "height": image.height},
                    "ocr_text": "",
                    "extraction_method": "single_image_no_text"
                }
            })
        
        return chunks

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
        logger.info("Enhanced OCR Agent started")
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