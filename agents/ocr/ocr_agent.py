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
        """Verbinde mit Redis mit korrekten Encoding-Einstellungen"""
        # KRITISCH: decode_responses=True für konsistentes String-Handling
        self.redis_client = await redis.from_url(
            self.redis_url,
            decode_responses=True,  # ✅ Automatisches Decoding von bytes zu strings
            encoding='utf-8',       # ✅ UTF-8 Encoding
            encoding_errors='strict' # ✅ Strikte Fehlerbehandlung
        )
        logger.info("OCR Agent connected to Redis with decode_responses=True")
    
    async def process_queue(self):
        """Verarbeite Processing-Queue mit robustem JSON Handling"""
        while self.running:
            try:
                # brpop gibt ein Tuple zurück: (queue_name, data)
                result = await self.redis_client.brpop('processing_queue', timeout=5)
                
                if result:
                    queue_name, task_data = result
                    logger.debug(f"Received from queue: type={type(task_data)}, data={task_data[:100]}...")
                    
                    try:
                        # Parse JSON - task_data ist bereits ein String dank decode_responses=True
                        task = json.loads(task_data)
                        
                        # Validiere erforderliche Felder
                        required_fields = ['task_id', 'file_path', 'task_type']
                        if not all(field in task for field in required_fields):
                            logger.error(f"Missing required fields in task: {task}")
                            continue
                            
                        if task['task_type'] == 'ocr':
                            await self.process_ocr_task(task)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        logger.error(f"Raw data: {repr(task_data)}")
                        # Optional: Sende fehlgeschlagene Tasks in eine Error-Queue
                        await self.redis_client.lpush('failed_json_tasks', task_data)
                        continue
                        
            except asyncio.TimeoutError:
                # Timeout ist normal bei brpop
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {str(e)}", exc_info=True)
                await asyncio.sleep(5)
    
    async def process_ocr_task(self, task: Dict[str, Any]):
        """Verarbeite OCR Task mit verbessertem Error Handling"""
        try:
            filepath = Path(task['file_path'])
            logger.info(f"Starting OCR for: {filepath}")
            
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                await self.handle_failed_task(task, 'File not found')
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
                
                # Sende Ergebnis zur Indexing-Queue
                await self.redis_client.lpush('indexing_queue', json.dumps(result))
                logger.info(f"OCR completed for: {filepath}")
            else:
                logger.warning(f"No text extracted from: {filepath}")
                await self.handle_failed_task(task, 'No text extracted')
                
        except Exception as e:
            logger.error(f"OCR error for {task['file_path']}: {str(e)}", exc_info=True)
            await self.handle_failed_task(task, str(e))
    
    async def handle_failed_task(self, task: Dict[str, Any], error_message: str):
        """Behandle fehlgeschlagene Tasks"""
        failed_task = {
            **task,
            'error': error_message,
            'failed_at': datetime.now().isoformat()
        }
        await self.redis_client.lpush('failed_tasks', json.dumps(failed_task))
    
    async def extract_text(self, filepath: Path) -> Optional[str]:
        """Extrahiere Text aus verschiedenen Dateiformaten"""
        ext = filepath.suffix.lower()
        
        try:
            if ext == '.pdf':
                return await self.ocr_pdf(filepath)
            elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                return await self.ocr_image(filepath)
            elif ext in ['.txt', '.md']:
                # Versuche verschiedene Encodings
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        return filepath.read_text(encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                logger.error(f"Could not decode text file: {filepath}")
                return None
            else:
                logger.warning(f"Unsupported format: {ext}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}", exc_info=True)
            return None
    
    async def ocr_pdf(self, filepath: Path) -> str:
        """OCR für PDF-Dateien"""
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
            logger.error(f"PDF OCR error: {str(e)}", exc_info=True)
            raise
    
    async def ocr_image(self, filepath: Path) -> str:
        """OCR für Bilddateien"""
        try:
            image = Image.open(filepath)
            
            text = pytesseract.image_to_string(
                image,
                lang=config.tesseract_lang
            )
            
            return text
            
        except Exception as e:
            logger.error(f"Image OCR error: {str(e)}", exc_info=True)
            raise
    
    async def run(self):
        """Hauptloop mit Graceful Shutdown"""
        await self.connect()
        logger.info("OCR Agent started")
        
        try:
            await self.process_queue()
        except KeyboardInterrupt:
            logger.info("OCR Agent shutting down gracefully")
            self.running = False
        finally:
            if self.redis_client:
                await self.redis_client.close()

def main():
    import os
    
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    agent = OCRAgent(redis_url)
    asyncio.run(agent.run())

if __name__ == "__main__":
    main()
