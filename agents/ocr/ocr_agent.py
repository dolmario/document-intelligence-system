import asyncio
import json
import pytesseract
from PIL import Image
import pdf2image
from pathlib import Path
import redis.asyncio as redis
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from core.utils import setup_logger, config
from core.models import ProcessingTask
from core.privacy import PrivacyManager

logger = setup_logger('ocr_agent', 'ocr.log')

class OCRAgent:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None
        self.privacy_manager = PrivacyManager()
        self.running = True
        
    async def connect(self):
        self.redis_client = await redis.from_url(self.redis_url)
        logger.info("OCR Agent connected to Redis")
    
    async def process_queue(self):
        while self.running:
            try:
                task_data = await self.redis_client.brpop('processing_queue', timeout=5)
                
                if task_data:
                    _, task_json = task_data
                    
                    try:
                        task = json.loads(task_json)
                        
                        required_fields = ['task_id', 'file_path', 'task_type']
                        if not all(field in task for field in required_fields):
                            logger.error(f"Missing required fields in task: {task}")
                            continue
                            
                        if task['task_type'] == 'ocr':
                            await self.process_ocr_task(task)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in queue: {task_json[:100]}... Error: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Task processing error: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Queue processing error: {str(e)}")
                await asyncio.sleep(5)
    
    async def process_ocr_task(self, task: Dict[str, Any]):
        try:
            filepath = Path(task['file_path'])
            logger.info(f"Starting OCR for: {filepath}")
            
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                await self.redis_client.lpush('failed_tasks', json.dumps({
                    **task,
                    'error': 'File not found',
                    'failed_at': datetime.now().isoformat()
                }))
                return
            
            text = await self.extract_text(filepath)
            
            if text:
                if config.enable_pii_removal:
                    text = self.privacy_manager.anonymize_text(text)
                
                result = {
                    'task_id': task['task_id'],
                    'file_path': str(filepath),
                    'extracted_text': text,
                    'ocr_completed_at': datetime.now().isoformat()
                }
                
                await self.redis_client.lpush('indexing_queue', json.dumps(result))
                logger.info(f"OCR completed for: {filepath}")
            else:
                logger.warning(f"No text extracted from: {filepath}")
                
        except Exception as e:
            logger.error(f"OCR error for {task['file_path']}: {str(e)}")
            await self.redis_client.lpush('failed_tasks', json.dumps({
                **task,
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }))
    
    async def extract_text(self, filepath: Path) -> Optional[str]:
        ext = filepath.suffix.lower()
        
        try:
            if ext == '.pdf':
                return await self.ocr_pdf(filepath)
            elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                return await self.ocr_image(filepath)
            elif ext in ['.txt', '.md']:
                try:
                    return filepath.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        return filepath.read_text(encoding='latin1')
                    except UnicodeDecodeError:
                        return filepath.read_text(encoding='cp1252')
            else:
                logger.warning(f"Unsupported format: {ext}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return None
    
    async def ocr_pdf(self, filepath: Path) -> str:
        try:
            images = pdf2image.convert_from_path(filepath, dpi=300)
            
            texts = []
            for i, image in enumerate(images):
                logger.debug(f"Processing page {i+1} of {len(images)}")
                text = pytesseract.image_to_string(
                    image, 
                    lang=config.tesseract_lang
                )
                texts.append(text)
            
            return '\n\n--- PAGE BREAK ---\n\n'.join(texts)
            
        except Exception as e:
            logger.error(f"PDF OCR error: {str(e)}")
            raise
    
    async def ocr_image(self, filepath: Path) -> str:
        try:
            image = Image.open(filepath)
            
            text = pytesseract.image_to_string(
                image,
                lang=config.tesseract_lang
            )
            
            return text
            
        except Exception as e:
            logger.error(f"Image OCR error: {str(e)}")
            raise
    
    async def run(self):
        await self.connect()
        logger.info("OCR Agent started")
        
        try:
            await self.process_queue()
        except KeyboardInterrupt:
            logger.info("OCR Agent shutting down")
            self.running = False

def main():
    import os
    
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    agent = OCRAgent(redis_url)
    asyncio.run(agent.run())

if __name__ == "__main__":
    main()
