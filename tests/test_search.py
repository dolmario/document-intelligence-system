import pytest
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from services.search.search_engine import IntelligentSearchEngine
from core.models import DocumentIndex

@pytest.fixture
def search_engine(tmp_path):
    """Search Engine Fixture mit temporärem Index-Pfad"""
    index_path = tmp_path / "indices"
    index_path.mkdir()
    (index_path / "json").mkdir()
    (index_path / "markdown").mkdir()
    
    engine = IntelligentSearchEngine(str(index_path))
    
    # Füge Test-Dokumente hinzu
    test_docs = [
        DocumentIndex(
            doc_id="doc1",
            doc_type="txt",
            title="Python Programmierung",
            content_hash="hash1",
            keywords=["python", "programmierung", "coding"],
            categories=["Technisch"],
            sections=[{"title": "Intro", "content": "Python ist eine Programmiersprache"}]
        ),
        DocumentIndex(
            doc_id="doc2",
            doc_type="pdf",
            title="Machine Learning Guide",
            content_hash="hash2",
            keywords=["machine", "learning", "ai", "python"],
            categories=["Technisch"],
            sections=[{"title": "ML Basics", "content": "Machine Learning mit Python"}]
        )
    ]
    
    for doc in test_docs:
        engine.indices[doc.doc_id] = doc
        doc.save(index_path)
    
    return engine

def test_keyword_search(search_engine):
    """Test Keyword-basierte Suche"""
    results = search_engine.keyword_search("python")
    
    assert len(results) == 2
    assert "doc1" in results
    assert "doc2" in results

def test_keyword_search_ranking(search_engine):
    """Test Ranking bei Keyword-Suche"""
    # Mock indices mit unterschiedlichen Scores
    results = search_engine.keyword_search("programmierung")
    
    # doc1 sollte höher gerankt sein (im Titel)
    assert results[0] == "doc1"

def test_semantic_search(search_engine):
    """Test semantische Suche"""
    candidates = ["doc1", "doc2"]
    results = search_engine.semantic_search("coding tutorial", candidates)
    
    assert len(results) == 2
    assert all(isinstance(r[1], float) for r in results)
    assert all(0 <= r[1] <= 1 for r in results)

def test_expand_with_links(search_engine):
    """Test Erweiterung mit Verknüpfungen"""
    # Füge Verknüpfung hinzu
    search_engine.indices["doc1"].references = ["doc3"]
    search_engine.indices["doc1"].semantic_links = {"doc2": 0.8}
    
    # Mock doc3
    search_engine.indices["doc3"] = DocumentIndex(
        doc_id="doc3",
        doc_type="txt",
        title="Related Document",
        content_hash="hash3"
    )
    
    results = [("doc1", 0.9)]
    expanded = search_engine.expand_with_links(results)
    
    assert len(expanded) > 1
    assert any(doc_id == "doc2" for doc_id, _ in expanded)
