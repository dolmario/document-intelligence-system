import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.ocr.ocr_agent import OCRAgent
from core.models import ProcessingTask

@pytest.fixture
def ocr_agent():
    """OCR Agent Fixture"""
    agent = OCRAgent("redis://localhost:6379")
    agent.redis_client = AsyncMock()
    return agent

@pytest.fixture
def sample_task():
    """Sample Task Fixture"""
    return {
        'task_id': 'test_123',
        'file_path': 'tests/fixtures/test.txt',
        'task_type': 'ocr',
        'priority': 5,
        'created_at': '2024-01-01T12:00:00'
    }

@pytest.mark.asyncio
async def test_extract_text_from_txt(ocr_agent):
    """Test Text-Extraktion aus TXT-Datei"""
    # Erstelle Test-Datei
    test_file = Path("tests/fixtures/test.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("Dies ist ein Test-Text")
    
    # Test
    result = await ocr_agent.extract_text(test_file)
    
    assert result == "Dies ist ein Test-Text"
    
    # Cleanup
    test_file.unlink()

@pytest.mark.asyncio
async def test_process_ocr_task(ocr_agent, sample_task):
    """Test OCR Task Processing"""
    # Mock extract_text
    ocr_agent.extract_text = AsyncMock(return_value="Extrahierter Text")
    
    # Test
    await ocr_agent.process_ocr_task(sample_task)
    
    # Assertions
    ocr_agent.redis_client.lpush.assert_called_once()
    call_args = ocr_agent.redis_client.lpush.call_args[0]
    assert call_args[0] == 'indexing_queue'
    
    import json
    result = json.loads(call_args[1])
    assert result['task_id'] == 'test_123'
    assert result['extracted_text'] == "Extrahierter Text"

@pytest.mark.asyncio
async def test_ocr_with_pii_removal(ocr_agent, sample_task):
    """Test OCR mit PII-Entfernung"""
    # Mock extract_text mit PII
    test_text = "Kontakt: max.mustermann@example.com oder +49 123 456789"
    ocr_agent.extract_text = AsyncMock(return_value=test_text)
    
    # Aktiviere PII-Entfernung
    from core.utils import config
    config.enable_pii_removal = True
    
    # Test
    await ocr_agent.process_ocr_task(sample_task)
    
    # Check ob PII entfernt wurde
    call_args = ocr_agent.redis_client.lpush.call_args[0]
    import json
    result = json.loads(call_args[1])
    
    assert "[EMAIL]" in result['extracted_text']
    assert "[PHONE]" in result['extracted_text']
    assert "max.mustermann@example.com" not in result['extracted_text']
