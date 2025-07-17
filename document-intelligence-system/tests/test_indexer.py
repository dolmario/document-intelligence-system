import pytest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.indexer.index_generator import DSGVOCompliantIndexer
from core.models import DocumentIndex

@pytest.fixture
def indexer():
    """Indexer Fixture"""
    return DSGVOCompliantIndexer()

def test_remove_pii(indexer):
    """Test PII-Entfernung"""
    text = """
    Sehr geehrter Max Mustermann,
    Ihre E-Mail max@example.com wurde empfangen.
    Telefon: +49 123 456789
    IBAN: DE89 3704 0044 0532 0130 00
    """
    
    result = indexer.remove_pii(text)
    
    assert "[EMAIL]" in result
    assert "[PHONE]" in result
    assert "[IBAN]" in result
    assert "max@example.com" not in result
    assert "+49 123 456789" not in result

def test_extract_keywords(indexer):
    """Test Keyword-Extraktion"""
    text = """
    Das Document Intelligence System verarbeitet Dokumente automatisch.
    Es nutzt OCR für Texterkennung und erstellt durchsuchbare Indizes.
    Die Verarbeitung erfolgt DSGVO-konform.
    """
    
    keywords = indexer.extract_keywords(text)
    
    assert len(keywords) > 0
    assert any("dokument" in kw.lower() for kw in keywords)
    assert any("ocr" in kw.lower() for kw in keywords)

def test_detect_references(indexer):
    """Test Referenz-Erkennung"""
    text = """
    Dieses Dokument basiert auf dem Konzeptpapier 2023.
    Siehe auch: Technische Dokumentation
    Vgl. Anhang A für weitere Details.
    """
    
    references = indexer.detect_references(text)
    
    assert len(references) >= 2
    assert all(isinstance(ref, str) for ref in references)

def test_auto_categorize(indexer):
    """Test Auto-Kategorisierung"""
    text = """
    RECHNUNG
    
    Rechnungsnummer: 2024-001
    Betrag: 1000 EUR zzgl. MwSt.
    """
    
    categories = indexer.auto_categorize(text, [])
    
    assert "Rechnung" in categories

def test_generate_index(indexer):
    """Test vollständige Index-Generierung"""
    content = """
    Testdokument für Index-Generierung
    
    Dies ist ein Test mit E-Mail: test@example.com
    """
    
    metadata = {
        'path': '/test/document.txt',
        'type': 'txt'
    }
    
    index = indexer.generate_index(content, metadata)
    
    assert isinstance(index, DocumentIndex)
    assert index.title != ""
    assert len(index.keywords) > 0
    assert "[EMAIL]" in str(index.sections)
