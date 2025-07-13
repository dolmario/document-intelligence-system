python# agents/watchdog/watchdog_agent.py
import asyncio
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Set, Dict
import redis.asyncio as redis
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import logging

from core.utils import setup_logger, config, get_file_extension
from core.models import ProcessingTask

logger = setup_logger('watchdog_agent', 'logs/watchdog.log')

class DocumentWatchdog(FileSystemEventHandler):
    """Überwacht Dateisystem auf neue/geänderte Dokumente"""
    
    def __init__(self, watch_paths: list, redis_url: str):
        self.watch_paths = [Path(p) for p in watch_paths]
        self.redis_url = redis_url
        self.redis_client = None
        self.file_cache: Dict[str, str] = {}  # filepath -> hash
        self.supported_extensions = {
            'pdf', 'txt', 'docx', 'doc', 'jpg', 'jpeg', 'png', 'tiff', 'tif'
        }
        self.loop = asyncio.new_event_loop()
        
    async def connect_redis(self):
        """Stelle Redis-Verbindung her"""
        self.redis_client = await redis.from_url(self.redis_url)
        logger.info("Redis-Verbindung hergestellt")
    
    def on_created(self, event: FileSystemEvent):
        """Neue Datei erkannt"""
        if not event.is_directory:
            self.loop.run_until_complete(self.process_new_file(event.src_path))
    
    def on_modified(self, event: FileSystemEvent):
        """Datei geändert"""
        if not event.is_directory:
            self.loop.run_until_complete(self.check_file_changes(event.src_path))
    
    def on_deleted(self, event: FileSystemEvent):
        """Datei gelöscht"""
        if not event.is_directory:
            self.loop.run_until_complete(self.mark_file_deleted(event.src_path))
    
    async def process_new_file(self, filepath: str):
        """Verarbeite neue Datei"""
        try:
            # Prüfe ob Datei unterstützt wird
            ext = get_file_extension(filepath)
            if ext not in self.supported_extensions:
                logger.debug(f"Ignoriere nicht unterstützte Datei: {filepath}")
                return
            
            # Berechne Hash
            file_hash = await self.calculate_file_hash(filepath)
            
            # Prüfe ob Datei bereits verarbeitet wurde
            if filepath in self.file_cache and self.file_cache[filepath] == file_hash:
                logger.debug(f"Datei bereits verarbeitet: {filepath}")
                return
            
            # Erstelle Task
            task = ProcessingTask(
                task_id=hashlib.sha256(f"{filepath}_{datetime.now().isoformat()}".encode()).hexdigest(),
                file_path=filepath,
                task_type='ocr' if ext in ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'tif'] else 'index',
                priority=5
            )
            
            # Füge zur Queue hinzu
            await self.redis_client.lpush('processing_queue', json.dumps({
                'task_id': task.task_id,
                'file_path': task.file_path,
                'task_type': task.task_type,
                'priority': task.priority,
                'created_at': task.created_at
            }))
            
            # Update Cache
            self.file_cache[filepath] = file_hash
            
            logger.info(f"Neue Datei zur Verarbeitung hinzugefügt: {filepath}")
            
        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten von {filepath}: {str(e)}")
    
    async def check_file_changes(self, filepath: str):
        """Prüfe ob Datei geändert wurde"""
        try:
            new_hash = await self.calculate_file_hash(filepath)
            
            if filepath in self.file_cache and self.file_cache[filepath] != new_hash:
                logger.info(f"Datei geändert: {filepath}")
                await self.process_new_file(filepath)
                
        except Exception as e:
            logger.error(f"Fehler beim Prüfen von Änderungen: {str(e)}")
    
    async def mark_file_deleted(self, filepath: str):
        """Markiere Datei als gelöscht"""
        try:
            if filepath in self.file_cache:
                del self.file_cache[filepath]
                
            # Informiere Index-System
            await self.redis_client.lpush('deletion_queue', json.dumps({
                'filepath': filepath,
                'deleted_at': datetime.now().isoformat()
            }))
            
            logger.info(f"Datei als gelöscht markiert: {filepath}")
            
        except Exception as e:
            logger.error(f"Fehler beim Markieren als gelöscht: {str(e)}")
    
    async def calculate_file_hash(self, filepath: str) -> str:
        """Berechne SHA256 Hash einer Datei"""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Fehler beim Hash-Berechnen für {filepath}: {str(e)}")
            return ""
    
    def start(self):
        """Starte Watchdog"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.connect_redis())
        
        observer = Observer()
        
        for path in self.watch_paths:
            if path.exists():
                observer.schedule(self, str(path), recursive=True)
                logger.info(f"Überwache Verzeichnis: {path}")
            else:
                logger.warning(f"Verzeichnis existiert nicht: {path}")
        
        observer.start()
        logger.info("Watchdog gestartet")
        
        try:
            while True:
                asyncio.run(asyncio.sleep(1))
        except KeyboardInterrupt:
            observer.stop()
            logger.info("Watchdog gestoppt")
        
        observer.join()

def main():
    """Hauptfunktion"""
    import os
    
    watch_paths = os.getenv('WATCH_PATHS', './data').split(',')
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    watchdog = DocumentWatchdog(watch_paths, redis_url)
    watchdog.start()

if __name__ == "__main__":
    main()
