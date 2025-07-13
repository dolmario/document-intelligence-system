python# test_pipeline.py
import asyncio
import redis.asyncio as redis
import json
from pathlib import Path

async def test_pipeline():
    """Teste die Document Processing Pipeline"""
    
    # Verbinde mit Redis
    r = await redis.from_url('redis://localhost:6379')
    
    # Simuliere neue Datei
    test_task = {
        'task_id': 'test_001',
        'file_path': './data/test.pdf',
        'task_type': 'ocr',
        'priority': 5,
        'created_at': '2024-01-01T12:00:00'
    }
    
    # Füge zur Queue hinzu
    await r.lpush('processing_queue', json.dumps(test_task))
    print(f"Task hinzugefügt: {test_task['task_id']}")
    
    # Überwache Queues
    print("\nÜberwache Queues...")
    for i in range(30):
        ocr_len = await r.llen('processing_queue')
        idx_len = await r.llen('indexing_queue')
        learn_len = await r.llen('learning_queue')
        
        print(f"OCR Queue: {ocr_len}, Index Queue: {idx_len}, Learning Queue: {learn_len}")
        
        await asyncio.sleep(2)
    
    await r.close()

if __name__ == "__main__":
    asyncio.run(test_pipeline())
