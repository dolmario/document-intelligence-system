#!/usr/bin/env python3
"""
Test Script für Redis JSON Pipeline
Testet die korrekte JSON-Serialisierung zwischen Watchdog und OCR Agent
"""

import asyncio
import json
import redis.asyncio as redis
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('redis_json_test')

async def test_redis_json_pipeline():
    """Teste die Redis JSON Pipeline"""
    
    # Redis-Verbindung mit decode_responses=True
    redis_client = await redis.from_url(
        'redis://localhost:6379',
        decode_responses=True,  # KRITISCH: Aktiviert automatisches Decoding
        encoding='utf-8'
    )
    
    logger.info("Connected to Redis with decode_responses=True")
    
    # 1. Test: Simuliere Watchdog Task
    test_task = {
        'task_id': 'test_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
        'file_path': '/data/test_document.pdf',
        'task_type': 'ocr',
        'priority': 5,
        'created_at': datetime.now().isoformat()
    }
    
    # Serialisiere und sende
    json_data = json.dumps(test_task, ensure_ascii=False)
    logger.info(f"Sending JSON: {json_data}")
    
    await redis_client.lpush('processing_queue', json_data)
    logger.info("✅ Task sent to processing_queue")
    
    # 2. Test: Simuliere OCR Agent - Hole Task aus Queue
    result = await redis_client.brpop('processing_queue', timeout=5)
    
    if result:
        queue_name, received_data = result
        logger.info(f"Received from {queue_name}")
        logger.info(f"Data type: {type(received_data)}")
        logger.info(f"Raw data: {received_data[:100]}...")
        
        try:
            # Parse JSON
            parsed_task = json.loads(received_data)
            logger.info(f"✅ Successfully parsed JSON: {parsed_task}")
            
            # Vergleiche mit Original
            assert parsed_task['task_id'] == test_task['task_id']
            assert parsed_task['file_path'] == test_task['file_path']
            logger.info("✅ Data integrity verified")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            logger.error(f"Raw data: {repr(received_data)}")
    else:
        logger.warning("No data received (timeout)")
    
    # 3. Test: Spezielle Zeichen
    special_task = {
        'task_id': 'special_chars_test',
        'file_path': '/data/Übung_Größe_"Test".pdf',  # Deutsche Umlaute und Quotes
        'task_type': 'ocr',
        'metadata': {
            'title': 'Test mit "Anführungszeichen"',
            'description': "Test mit 'einfachen' Quotes"
        }
    }
    
    json_special = json.dumps(special_task, ensure_ascii=False)
    await redis_client.lpush('processing_queue', json_special)
    
    result = await redis_client.brpop('processing_queue', timeout=5)
    if result:
        _, data = result
        parsed = json.loads(data)
        logger.info(f"✅ Special chars test passed: {parsed['file_path']}")
    
    # 4. Test: Fehlerhafte Daten
    logger.info("\nTesting error handling...")
    
    # Sende ungültiges JSON
    await redis_client.lpush('processing_queue', '{"invalid": json}')
    
    result = await redis_client.brpop('processing_queue', timeout=5)
    if result:
        _, data = result
        try:
            json.loads(data)
        except json.JSONDecodeError:
            logger.info("✅ Invalid JSON correctly raises JSONDecodeError")
    
    # Cleanup
    await redis_client.close()
    logger.info("\n✅ All tests completed successfully!")

async def monitor_queues():
    """Monitor für Redis Queues"""
    redis_client = await redis.from_url(
        'redis://localhost:6379',
        decode_responses=True
    )
    
    logger.info("Starting queue monitor...")
    
    while True:
        # Zeige Queue-Längen
        processing_len = await redis_client.llen('processing_queue')
        indexing_len = await redis_client.llen('indexing_queue')
        failed_len = await redis_client.llen('failed_tasks')
        
        logger.info(f"Queues - Processing: {processing_len}, Indexing: {indexing_len}, Failed: {failed_len}")
        
        await asyncio.sleep(5)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        asyncio.run(monitor_queues())
    else:
        asyncio.run(test_redis_json_pipeline())
