import pytest
import asyncio
import redis.asyncio as redis
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline():
    """Test der vollständigen Pipeline"""
    # Redis Verbindung
    r = await redis.from_url("redis://localhost:6379")
    
    try:
        # Cleanup
        await r.flushdb()
        
        # Simuliere neue Datei
        test_task = {
            'task_id': 'integration_test_001',
            'file_path': 'tests/fixtures/test_integration.txt',
            'task_type': 'ocr',
            'priority': 10,
            'created_at': '2024-01-01T12:00:00'
        }
        
        # Erstelle Test-Datei
        test_file = Path(test_task['file_path'])
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("Integration Test Document")
        
        # Füge Task zur Queue
        await r.lpush('processing_queue', json.dumps(test_task))
        
        # Verifiziere Queue
        queue_length = await r.llen('processing_queue')
        assert queue_length == 1
        
        # Hole Task
        task_data = await r.rpop('processing_queue')
        assert task_data is not None
        
        retrieved_task = json.loads(task_data)
        assert retrieved_task['task_id'] == 'integration_test_001'
        
        # Cleanup
        test_file.unlink()
        
    finally:
        await r.close()

@pytest.mark.integration
def test_docker_compose_config():
    """Test Docker Compose Konfiguration"""
    import yaml
    
    with open('docker-compose.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Verifiziere Services
    required_services = ['redis', 'watchdog', 'ocr_agent', 'indexer', 'search_api', 'n8n', 'ollama']
    
    for service in required_services:
        assert service in config['services'], f"Service {service} fehlt in docker-compose.yml"
    
    # Verifiziere Netzwerk
    assert 'doc_network' in config['networks']
    
    # Verifiziere Volumes
    assert 'redis_data' in config['volumes']
