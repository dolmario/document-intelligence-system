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

# FIXED: Korrigierter Logger-Aufruf ohne doppelten logs-Pfad
logger = setup_logger('ocr_agent', 'ocr.log')

class OCRAgent:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None
        self.privacy_manager = PrivacyManager()
        self.running = True
        
    async def extract_docx(self, filepath: Path) -> str:
        """Extrahiere Text aus DOCX-Datei"""
        try:
            from docx import Document
            doc = Document(filepath)
            
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"DOCX Extraktion Fehler: {str(e)}")
            return ""
    
    async def run(self):
        """Hauptloop"""
        await self.connect()
        logger.info("OCR Agent gestartet")
        
        try:
            await self.process_queue()
        except KeyboardInterrupt:
            logger.info("OCR Agent wird beendet")
            self.running = False
        finally:
            if self.redis_client:
                await self.redis_client.close()

def main():
    import os
    
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    agent = OCRAgent(redis_url)
    
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logger.info("OCR Agent durch Benutzer beendet")
    except Exception as e:
        logger.error(f"OCR Agent Fehler: {str(e)}")

if __name__ == "__main__":
    main() connect(self):
        """Verbinde mit Redis"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            # Test der Verbindung
            await self.redis_client.ping()
            logger.info("OCR Agent verbunden mit Redis")
        except Exception as e:
            logger.error(f"Redis-Verbindung fehlgeschlagen: {str(e)}")
            raise
    
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
                        
            except asyncio.TimeoutError:
                # Normal bei timeout - einfach weiter
                continue
            except json.JSONDecodeError as e:
                logger.error(f"JSON Decode Fehler: {str(e)}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Fehler in process_queue: {str(e)}")
                await asyncio.sleep(5)
    
    async def process_ocr_task(self, task: Dict[str, Any]):
        """Führe OCR für eine Datei durch"""
        try:
            filepath = Path(task['file_path'])
            logger.info(f"Starte OCR für: {filepath}")
            
            # Prüfe ob Datei existiert
            if not filepath.exists():
                logger.warning(f"Datei existiert nicht: {filepath}")
                return
            
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
                    'ocr_completed_at': datetime.now().isoformat(),
                    'text_length': len(text),
                    'processing_time': task.get('processing_time', 0)
                }
                
                # An Indexer-Queue senden
                await self.redis_client.lpush('indexing_queue', json.dumps(result))
                logger.info(f"OCR abgeschlossen für: {filepath} ({len(text)} Zeichen)")
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
            elif ext == '.docx':
                return await self.extract_docx(filepath)
            else:
                logger.warning(f"Nicht unterstütztes Format: {ext}")
                return None
                
        except Exception as e:
            logger.error(f"Fehler beim Text-Extraktion: {str(e)}")
            return None
    
    async def ocr_pdf(self, filepath: Path) -> str:
        """OCR für PDF"""
        try:
            # PDF zu Bildern konvertieren - mit besseren Einstellungen
            images = pdf2image.convert_from_path(
                filepath, 
                dpi=300,
                first_page=1,
                last_page=None,
                fmt='png'
            )
            
            texts = []
            for i, image in enumerate(images):
                logger.debug(f"Verarbeite Seite {i+1} von {len(images)}")
                
                # OCR mit besseren Parametern
                text = pytesseract.image_to_string(
                    image, 
                    lang=config.tesseract_lang,
                    config='--oem 3 --psm 6'  # Bessere OCR-Parameter
                )
                
                if text.strip():  # Nur nicht-leere Seiten hinzufügen
                    texts.append(f"--- SEITE {i+1} ---\n{text}")
            
            return '\n\n'.join(texts)
            
        except Exception as e:
            logger.error(f"PDF OCR Fehler: {str(e)}")
            raise
    
    async def ocr_image(self, filepath: Path) -> str:
        """OCR für Bilder"""
        try:
            image = Image.open(filepath)
            
            # Bildvorverarbeitung für bessere OCR-Ergebnisse
            # Konvertiere zu RGB falls nötig
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # OCR mit optimierten Parametern
            text = pytesseract.image_to_string(
                image,
                lang=config.tesseract_lang,
                config='--oem 3 --psm 6'
            )
            
            return text
            
        except Exception as e:
            logger.error(f"Bild OCR Fehler: {str(e)}")
            raise
    
    async def
