import asyncio
import json
import base64
import io
from typing import Dict, Any, Optional
import pytesseract
from PIL import Image
import pdf2image
import redis.asyncio as redis
import logging
from datetime import datetime
from pathlib import Path
import os

# Setup logging properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ocr_agent')

class OCRAgent:
    """OCR Agent with N8N integration and proper file handling"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None
        self.running = True
        
    async def connect(self):
        """Connect to Redis with consistent settings"""
        self.redis_client = await redis.from_url(
            self.redis_url,
            decode_responses=True,  # Critical: Enable automatic decoding
            encoding='utf-8',
            encoding_errors='strict'
        )
        logger.info("OCR Agent connected to Redis with decode_responses=True")
    
    async def process_queue(self):
        """Process OCR tasks from queue"""
        while self.running:
            try:
                result = await self.redis_client.brpop('processing_queue', timeout=5)
                
                if result:
                    _, task_data = result
                    logger.debug(f"Received task data type: {type(task_data)}")
                    logger.debug(f"Task data preview: {task_data[:100] if isinstance(task_data, str) else task_data}")
                    
                    # Parse JSON
                    task = json.loads(task_data)
                    
                    # Process based on task type
                    if 'file_content' in task and 'file_name' in task:
                        # N8N task with base64 content
                        await self.process_n8n_task(task)
                    elif 'file_path' in task:
                        # File path based task
                        await self.process_file_task(task)
                    else:
                        logger.error(f"Invalid task format: {task}")
                        
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.error(f"Raw data: {task_data if 'task_data' in locals() else 'No data'}")
            except Exception as e:
                logger.error(f"Queue processing error: {str(e)}", exc_info=True)
                await asyncio.sleep(5)
    
    async def process_file_task(self, task: Dict[str, Any]):
        """Process file-based task"""
        try:
            file_path = task['file_path']
            logger.info(f"Processing file task: {file_path}")
            
            # Handle both absolute and relative paths
            if not os.path.isabs(file_path):
                # Try common mount points
                possible_paths = [
                    file_path,
                    f"/data/{os.path.basename(file_path)}",
                    f"/app/data/{os.path.basename(file_path)}"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        file_path = path
                        break
                else:
                    logger.error(f"File not found in any location: {task['file_path']}")
                    logger.info(f"Checked paths: {possible_paths}")
                    logger.info(f"Available files in /data: {os.listdir('/data') if os.path.exists('/data') else 'Directory not found'}")
                    return
            
            # Read file
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Extract text
            text = await self.extract_text_from_bytes(
                file_content,
                os.path.basename(file_path),
                self.get_mime_type(file_path)
            )
            
            await self.send_to_indexing(task, text)
            
        except Exception as e:
            logger.error(f"File task processing error: {str(e)}", exc_info=True)
    
    async def process_n8n_task(self, task: Dict[str, Any]):
        """Process N8N task with base64 content"""
        try:
            logger.info(f"Processing N8N task: {task['task_id']} - {task['file_name']}")
            
            # Decode base64 content
            file_content = base64.b64decode(task['file_content'])
            
            # Extract text based on task type
            if task.get('task_type') == 'text':
                text = file_content.decode('utf-8', errors='ignore')
            else:
                text = await self.extract_text_from_bytes(
                    file_content,
                    task['file_name'],
                    task.get('mime_type', 'application/octet-stream')
                )
            
            await self.send_to_indexing(task, text)
            
        except Exception as e:
            logger.error(f"N8N task processing error: {str(e)}", exc_info=True)
    
    async def extract_text_from_bytes(self, file_bytes: bytes, filename: str, mime_type: str) -> str:
        """Extract text from file bytes"""
        try:
            # PDF
            if mime_type == 'application/pdf' or filename.lower().endswith('.pdf'):
                return await self.ocr_pdf_from_bytes(file_bytes)
            
            # Images
            elif mime_type.startswith('image/') or filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                return await self.ocr_image_from_bytes(file_bytes)
            
            # Text files
            elif mime_type.startswith('text/') or filename.lower().endswith(('.txt', '.md', '.csv')):
                return file_bytes.decode('utf-8', errors='ignore')
            
            else:
                logger.warning(f"Unsupported file type: {mime_type} / {filename}")
                return ""
                
        except Exception as e:
            logger.error(f"Text extraction error: {str(e)}", exc_info=True)
            return ""
    
    async def ocr_pdf_from_bytes(self, pdf_bytes: bytes) -> str:
        """OCR for PDF from bytes"""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_bytes(pdf_bytes, dpi=300)
            
            texts = []
            for i, image in enumerate(images):
                logger.debug(f"Processing PDF page {i+1}")
                text = pytesseract.image_to_string(
                    image,
                    lang=os.getenv('TESSERACT_LANG', 'deu+eng')
                )
                texts.append(text)
            
            return '\n\n--- PAGE BREAK ---\n\n'.join(texts)
            
        except Exception as e:
            logger.error(f"PDF OCR error: {str(e)}", exc_info=True)
            raise
    
    async def ocr_image_from_bytes(self, image_bytes: bytes) -> str:
        """OCR for image from bytes"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            text = pytesseract.image_to_string(
                image,
                lang=os.getenv('TESSERACT_LANG', 'deu+eng')
            )
            
            return text
            
        except Exception as e:
            logger.error(f"Image OCR error: {str(e)}", exc_info=True)
            raise
    
    async def send_to_indexing(self, task: Dict[str, Any], text: str):
        """Send extracted text to indexing queue"""
        if not text:
            logger.warning(f"No text extracted from: {task.get('file_name', task.get('file_path', 'unknown'))}")
            return
        
        result = {
            'task_id': task['task_id'],
            'file_path': task.get('file_path', f"/data/{task.get('file_name', 'unknown')}"),
            'file_name': task.get('file_name', os.path.basename(task.get('file_path', 'unknown'))),
            'source': task.get('source', 'unknown'),
            'extracted_text': text,
            'metadata': task.get('metadata', {}),
            'ocr_completed_at': datetime.now().isoformat()
        }
        
        # Send to indexing queue
        await self.redis_client.lpush('indexing_queue', json.dumps(result, ensure_ascii=False))
        logger.info(f"OCR completed and sent to indexing: {result['file_name']}")
    
    def get_mime_type(self, file_path: str) -> str:
        """Get MIME type from file extension"""
        ext = Path(file_path).suffix.lower()
        mime_types = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }
        return mime_types.get(ext, 'application/octet-stream')
    
    async def run(self):
        """Main loop"""
        await self.connect()
        logger.info("OCR Agent started")
        
        try:
            await self.process_queue()
        except KeyboardInterrupt:
            logger.info("OCR Agent shutting down")
            self.running = False
        finally:
            if self.redis_client:
                await self.redis_client.close()

def main():
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    agent = OCRAgent(redis_url)
    asyncio.run(agent.run())

if __name__ == "__main__":
    main()
