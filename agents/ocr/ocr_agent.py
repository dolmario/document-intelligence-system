import asyncio
import json
import pytesseract
from PIL import Image
import pdf2image
from pathlib import Path
import redis.asyncio as redis
from typing import Optional, Dict, Any
import logging
from datetime import datetime  # ADDED

from core.utils import setup_logger, config
from core.models import ProcessingTask
from core.privacy import PrivacyManager

logger = setup_logger('ocr_agent', 'logs/ocr.log')

class OCRAgent:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None
        self.privacy_manager = PrivacyManager()
        self.running = True
        
    async def connect(self):
        """Verbinde mit Redis"""
        self.redis_client = await redis.from_url(self.redis_url)
        logger.info("OCR Agent verbunden mit Redis")
    
    async def process_queue(self):
        """Verarbeite OCR-Queue"""
        while self.running:
            try:
                # Hole Task aus Queue (blockierend mit Timeout)
                task_data = await self.redis_client.brpop('processing_queue', timeout=5)
                
                if task_data:
                    _, task_json = task_data
                    task = json.loads(task_json)
                    
                    if task['task_type'] == 'ocr':
                        await self.process_ocr_task(task)
                        
            except Exception as e:
                logger.error(f"Fehler in process_queue: {str(e)}")
                await asyncio.sleep(5)
    
    async def process_ocr_task(self, task: Dict[str, Any]):
        """Führe OCR für eine Datei durch"""
        try:
            filepath = Path(task['file_path'])
            logger.info(f"Starte OCR für: {filepath}")
            
            # OCR durchführen
            text = await self.extract_text(filepath)
            
            if text:
                # PII entfernen wenn aktiviert
                if config.enable_pii_removal:
                    text = self.privacy_manager.anonymize_text(text)
                
                # Ergebnis für Indexer vorbereiten
                result = {
                    'task_id': task['task_id'],
                    'file_path': str(filepath),
                    'extracted_text': text,
                    'ocr_completed_at': datetime.now().isoformat()
                }
                
                # An Indexer-Queue senden
                await self.redis_client.lpush('indexing_queue', json.dumps(result))
                logger.info(f"OCR abgeschlossen für: {filepath}")
            else:
                logger.warning(f"Kein Text extrahiert aus: {filepath}")
                
        except Exception as e:
            logger.error(f"OCR Fehler für {task['file_path']}: {str(e)}")
            # Task als fehlgeschlagen markieren
            await self.redis_client.lpush('failed_tasks', json.dumps({
                **task,
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }))
    
    async def extract_text(self, filepath: Path) -> Optional[str]:
        """Extrahiere Text aus Datei"""
        ext = filepath.suffix.lower()
        
        try:
            if ext == '.pdf':
                return await self.ocr_pdf(filepath)
            elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                return await self.ocr_image(filepath)
            elif ext in ['.txt', '.md']:
                return filepath.read_text(encoding='utf-8')
            else:
                logger.warning(f"Nicht unterstütztes Format: {ext}")
                return None
                
        except Exception as e:
            logger.error(f"Fehler beim Text-Extraktion: {str(e)}")
            return None
    
    async def ocr_pdf(self, filepath: Path) -> str:
        """OCR für PDF"""
        try:
            # PDF zu Bildern konvertieren
            images = pdf2image.convert_from_path(filepath, dpi=300)
            
            texts = []
            for i, image in enumerate(images):
                logger.debug(f"Verarbeite Seite {i+1} von {len(images)}")
                text = pytesseract.image_to_string(
                    image, 
                    lang=config.tesseract_lang
                )
                texts.append(text)
            
            return '\n\n--- SEITE ---\n\n'.join(texts)
            
        except Exception as e:
            logger.error(f"PDF OCR Fehler: {str(e)}")
            raise
    
    async def ocr_image(self, filepath: Path) -> str:
        """OCR für Bilder"""
        try:
            image = Image.open(filepath)
            
            # Preprocessing für bessere OCR-Ergebnisse
            # (könnte erweitert werden mit mehr Bildverarbeitung)
            
            text = pytesseract.image_to_string(
                image,
                lang=config.tesseract_lang
            )
            
            return text
            
        except Exception as e:
            logger.error(f"Bild OCR Fehler: {str(e)}")
            raise
    
    async def run(self):
        """Hauptloop"""
        await self.connect()
        logger.info("OCR Agent gestartet")
        
        try:
            await self.process_queue()
        except KeyboardInterrupt:
            logger.info("OCR Agent wird beendet")
            self.running = False

def main():
    import os
    from datetime import datetime
    
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    agent = OCRAgent(redis_url)
    asyncio.run(agent.run())

if __name__ == "__main__":
    main()
