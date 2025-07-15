import asyncio
import json
from pathlib import Path
import redis.asyncio as redis
from typing import Dict, Any
import logging

from core.utils import setup_logger, config
from core.models import DocumentIndex
from core.privacy import PrivacyManager
from agents.indexer.index_generator import DSGVOCompliantIndexer

logger = setup_logger('indexer_agent', 'indexer.log')

class IndexerAgent:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None
        self.indexer = DSGVOCompliantIndexer()
        self.running = True
        
    async def connect(self):
        """Verbinde mit Redis"""
        self.redis_client = await redis.from_url(self.redis_url)
        logger.info("Indexer Agent verbunden mit Redis")
    
    async def process_queue(self):
        """Verarbeite Indexing-Queue"""
        while self.running:
            try:
                # Hole Task aus Queue
                task_data = await self.redis_client.brpop('indexing_queue', timeout=5)
                
                if task_data:
                    _, task_json = task_data
                    task = json.loads(task_json)
                    await self.create_index(task)
                    
            except Exception as e:
                logger.error(f"Fehler in process_queue: {str(e)}")
                await asyncio.sleep(5)
    
    async def create_index(self, task: Dict[str, Any]):
        """Erstelle Index für Dokument"""
        try:
            logger.info(f"Erstelle Index für: {task['file_path']}")
            
            # Metadata vorbereiten
            metadata = {
                'path': task['file_path'],
                'type': Path(task['file_path']).suffix.lower().lstrip('.'),
                'ocr_task_id': task.get('task_id', '')
            }
            
            # Index generieren
            content = task.get('extracted_text', '')
            if not content:
                logger.warning(f"Kein Inhalt für Indexierung: {task['file_path']}")
                return
            
            index = self.indexer.generate_index(content, metadata)
            
            # Index speichern
            index.save(config.index_path)
            
            # Benachrichtigung für Learning Agent
            await self.redis_client.lpush('learning_queue', json.dumps({
                'doc_id': index.doc_id,
                'action': 'new_document',
                'timestamp': index.indexed_at
            }))
            
            logger.info(f"Index erstellt: {index.doc_id[:8]}... für {task['file_path']}")
            
        except Exception as e:
            logger.error(f"Indexierung fehlgeschlagen für {task['file_path']}: {str(e)}")
    
    async def run(self):
        """Hauptloop"""
        await self.connect()
        logger.info("Indexer Agent gestartet")
        
        try:
            await self.process_queue()
        except KeyboardInterrupt:
            logger.info("Indexer Agent wird beendet")
            self.running = False

def main():
    import os
    
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    agent = IndexerAgent(redis_url)
    asyncio.run(agent.run())

if __name__ == "__main__":
    main()
