import pytest
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Mock für SentenceTransformer, um Importprobleme zu vermeiden
class MockSentenceTransformer:
    def encode(self, text):
        return [0.1, 0.2, 0.3]  # Dummy-Embedding

@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", MockSentenceTransformer)

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
            title="Python Programmierung",
            keywords=["python", "programmierung", "coding"],
            sections=[{"title": "Intro", "content": "Python ist eine Programmiersprache"}],
            references=[],
            semantic_links={}
        ),
        DocumentIndex(
            doc_id="doc2",
            title="Machine Learning Guide",
            keywords=["machine", "learning", "ai", "python"],
            sections=[{"title": "ML Basics", "content": "Machine Learning mit Python"}],
            references=[],
            semantic_links={}
        )
    ]
    
    for doc in test_docs:
        engine.indices[doc.doc_id] = doc
        # Simuliere das Speichern des Index
        (index_path / "json" / f"{doc.doc_id}.json").touch()
    
    return engine

def test_keyword_search(search_engine):
    """Test Keyword-basierte Suche"""
    results = search_engine.keyword_search("python")
    
    assert len(results) == 2
    assert "doc1" in results
    assert "doc2" in results

def test_keyword_search_ranking(search_engine):
    """Test Ranking bei Keyword-Suche"""
    results = search_engine.keyword_search("programmierung")
    assert results[0] == "doc1"  # doc1 sollte höher gerankt sein

def test_semantic_search(search_engine):
    """Test semantische Suche"""
    candidates = ["doc1", "doc2"]
    results = search_engine.semantic_search("coding tutorial", candidates)
    assert len(results) == 2
    assert all(isinstance(r[1], float) for r in results)

def test_expand_with_links(search_engine):
    """Test Erweiterung mit Verknüpfungen"""
    search_engine.indices["doc1"].references = ["doc3"]
    search_engine.indices["doc1"].semantic_links = {"doc2": 0.8}
    
    # Füge doc3 hinzu
    search_engine.indices["doc3"] = DocumentIndex(
        doc_id="doc3",
        title="Related Document",
        keywords=[],
        sections=[],
        references=[],
        semantic_links={}
    )
    
    results = [("doc1", 0.9)]
    expanded = search_engine.expand_with_links(results)
    expanded_ids = [doc_id for doc_id, _ in expanded]
    
    assert len(expanded) == 3
    assert "doc2" in expanded_ids
    assert "doc3" in expanded_ids
