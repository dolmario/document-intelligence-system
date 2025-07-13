# tests/test_search.py
import pytest
from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).parent.parent))

# Mock für DocumentIndex
class MockDocumentIndex:
    def __init__(self, doc_id, title, keywords, sections, references=None, semantic_links=None):
        self.doc_id = doc_id
        self.title = title
        self.keywords = keywords
        self.sections = sections
        self.references = references or []
        self.semantic_links = semantic_links or {}
    
    def save(self, path):
        pass

@pytest.fixture
def search_engine(tmp_path):
    """Search Engine Fixture mit temporärem Index-Pfad"""
    from services.search.search_engine import IntelligentSearchEngine
    
    index_path = tmp_path / "indices"
    index_path.mkdir()
    (index_path / "json").mkdir()
    
    engine = IntelligentSearchEngine(str(index_path))
    engine.indices = {}
    
    # Füge Test-Dokumente hinzu
    test_docs = [
        MockDocumentIndex(
            doc_id="doc1",
            title="Python Programmierung",
            keywords=["python", "programmierung", "coding"],
            sections=[{"title": "Intro", "content": "Python ist eine Programmiersprache"}]
        ),
        MockDocumentIndex(
            doc_id="doc2",
            title="Machine Learning Guide",
            keywords=["machine", "learning", "ai", "python"],
            sections=[{"title": "ML Basics", "content": "Machine Learning mit Python"}]
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
    # doc1 sollte höher gerankt sein (Keyword im Titel)
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
    
    # Füge doc3 hinzu
    search_engine.indices["doc3"] = MockDocumentIndex(
        doc_id="doc3",
        title="Related Document",
        keywords=[],
        sections=[]
    )
    
    results = [("doc1", 0.9)]
    expanded = search_engine.expand_with_links(results)
    expanded_ids = [doc_id for doc_id, _ in expanded]
    
    assert len(expanded) == 3
    assert "doc2" in expanded_ids
    assert "doc3" in expanded_ids
