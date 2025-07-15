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

logger = logging.getLogger('n8n_ocr_agent')

class N8NBasedOCRAgent:
    """OCR Agent der mit N8N-übergebenen Base64-Daten arbeitet"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None
        self.running = True
        
    async def connect(self):
        """Verbinde mit Redis"""
        self.redis_client = await redis.from_url(
            self.redis_url,
            decode_responses=True,
            encoding='utf-8'
        )
        logger.info("N8N OCR Agent connected to Redis")
    
    async def process_queue(self):
        """Verarbeite OCR Tasks von N8N"""
        while self.running:
            try:
                result = await self.redis_client.brpop('processing_queue', timeout=5)
                
                if result:
                    _, task_data = result
                    task = json.loads(task_data)
                    
                    # Task enthält jetzt Base64-kodierten Dateiinhalt
                    if 'file_content' in task:
                        await self.process_n8n_task(task)
                    else:
                        # Fallback für alte Tasks
                        await self.process_legacy_task(task)
                        
            except Exception as e:
                logger.error(f"Queue processing error: {str(e)}", exc_info=True)
                await asyncio.sleep(5)
    
    async def process_n8n_task(self, task: Dict[str, Any]):
        """Verarbeite Task mit Base64-Content von N8N"""
        try:
            logger.info(f"Processing N8N task: {task['task_id']} - {task['file_name']}")
            
            # Dekodiere Base64-Content
            file_content = base64.b64decode(task['file_content'])
            
            # Bestimme Verarbeitungstyp
            if task['task_type'] == 'ocr':
                text = await self.extract_text_from_bytes(
                    file_content,
                    task['file_name'],
                    task.get('mime_type', 'application/octet-stream')
                )
            elif task['task_type'] == 'text':
                # Direkte Text-Verarbeitung
                text = file_content.decode('utf-8', errors='ignore')
            else:
                logger.warning(f"Unknown task type: {task['task_type']}")
                text = ""
            
            if text:
                # Erstelle Ergebnis
                result = {
                    'task_id': task['task_id'],
                    'file_name': task['file_name'],
                    'source': task.get('source', 'n8n'),
                    'extracted_text': text,
                    'metadata': task.get('metadata', {}),
                    'ocr_completed_at': datetime.now().isoformat()
                }
                
                # Sende zur Indexierung
                await self.redis_client.lpush('indexing_queue', json.dumps(result))
                logger.info(f"OCR completed for: {task['file_name']}")
                
                # Optional: Sende Erfolg-Notification an N8N
                await self.send_n8n_callback(task['task_id'], 'success', result)
                
            else:
                logger.warning(f"No text extracted from: {task['file_name']}")
                await self.send_n8n_callback(task['task_id'], 'failed', {'error': 'No text extracted'})
                
        except Exception as e:
            logger.error(f"N8N task processing error: {str(e)}", exc_info=True)
            await self.send_n8n_callback(task['task_id'], 'error', {'error': str(e)})
    
    async def extract_text_from_bytes(self, file_bytes: bytes, filename: str, mime_type: str) -> str:
        """Extrahiere Text aus Bytes"""
        try:
            # PDF
            if mime_type == 'application/pdf' or filename.lower().endswith('.pdf'):
                return await self.ocr_pdf_from_bytes(file_bytes)
            
            # Bilder
            elif mime_type.startswith('image/') or filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                return await self.ocr_image_from_bytes(file_bytes)
            
            # Text-Dateien
            elif mime_type.startswith('text/') or filename.lower().endswith(('.txt', '.md', '.csv')):
                return file_bytes.decode('utf-8', errors='ignore')
            
            else:
                logger.warning(f"Unsupported file type: {mime_type} / {filename}")
                return ""
                
        except Exception as e:
            logger.error(f"Text extraction error: {str(e)}", exc_info=True)
            return ""
    
    async def ocr_pdf_from_bytes(self, pdf_bytes: bytes) -> str:
        """OCR für PDF aus Bytes"""
        try:
            # Konvertiere PDF zu Bildern
            images = pdf2image.convert_from_bytes(pdf_bytes, dpi=300)
            
            texts = []
            for i, image in enumerate(images):
                logger.debug(f"Processing PDF page {i+1}")
                text = pytesseract.image_to_string(
                    image,
                    lang='deu+eng'  # Konfigurierbar machen
                )
                texts.append(text)
            
            return '\n\n--- PAGE BREAK ---\n\n'.join(texts)
            
        except Exception as e:
            logger.error(f"PDF OCR error: {str(e)}", exc_info=True)
            raise
    
    async def ocr_image_from_bytes(self, image_bytes: bytes) -> str:
        """OCR für Bild aus Bytes"""
        try:
            # Öffne Bild aus Bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            text = pytesseract.image_to_string(
                image,
                lang='deu+eng'  # Konfigurierbar machen
            )
            
            return text
            
        except Exception as e:
            logger.error(f"Image OCR error: {str(e)}", exc_info=True)
            raise
    
    async def send_n8n_callback(self, task_id: str, status: str, data: dict):
        """Sende Status-Update zurück an N8N"""
        callback_data = {
            'task_id': task_id,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Sende an spezielle Callback-Queue für N8N
        await self.redis_client.lpush('n8n_callbacks', json.dumps(callback_data))
    
    async def process_legacy_task(self, task: Dict[str, Any]):
        """Fallback für alte file_path basierte Tasks"""
        logger.warning(f"Legacy task received: {task.get('file_path', 'unknown')}")
        # Konvertiere zu N8N-Format wenn möglich
        # Oder ignoriere
        pass
    
    async def run(self):
        """Hauptloop"""
        await self.connect()
        logger.info("N8N-based OCR Agent started")
        
        try:
            await self.process_queue()
        except KeyboardInterrupt:
            logger.info("OCR Agent shutting down")
            self.running = False
        finally:
            if self.redis_client:
                await self.redis_client.close()

def main():
    import os
    
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    agent = N8NBasedOCRAgent(redis_url)
    asyncio.run(agent.run())

if __name__ == "__main__":
    main()
