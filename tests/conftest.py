import pytest
import asyncio
from pathlib import Path
import shutil
import sys
from unittest.mock import MagicMock

# Mock für sentence_transformers und huggingface_hub, bevor andere Importe erfolgen
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["sentence_transformers"].SentenceTransformer = MagicMock()
sys.modules["sentence_transformers"].SentenceTransformer.return_value.encode.return_value = [0.1] * 384

sys.modules["huggingface_hub"] = MagicMock()
sys.modules["huggingface_hub"].cached_download = MagicMock()
sys.modules["huggingface_hub"].hf_hub_url = MagicMock()

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path_factory):
    """Setup Test Environment"""
    # Erstelle temporäre Verzeichnisse
    test_root = tmp_path_factory.mktemp("test_run")
    
    for dir_name in ["data", "indices", "logs"]:
        (test_root / dir_name).mkdir()
    
    # Setze Umgebungsvariablen
    import os
    os.environ["DATA_PATH"] = str(test_root / "data")
    os.environ["INDEX_PATH"] = str(test_root / "indices")
    os.environ["REDIS_URL"] = "redis://localhost:6379"
    os.environ["ENABLE_PII_REMOVAL"] = "true"
    
    yield test_root
    
    # Cleanup
    shutil.rmtree(test_root, ignore_errors=True)

@pytest.fixture
def sample_documents():
    """Sample Dokumente für Tests"""
    return [
        {
            "content": "Dies ist ein Test-Dokument über Python Programmierung.",
            "metadata": {"path": "/test/doc1.txt", "type": "txt"}
        },
        {
            "content": "Machine Learning und Künstliche Intelligenz sind wichtige Themen.",
            "metadata": {"path": "/test/doc2.pdf", "type": "pdf"}
        },
        {
            "content": "RECHNUNG Nr. 2024-001\nKunde: Max Mustermann\nBetrag: 1000 EUR",
            "metadata": {"path": "/test/invoice.pdf", "type": "pdf"}
        }
    ]

# Marker für verschiedene Test-Typen
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
