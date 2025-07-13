# services/search/api.py
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
from pathlib import Path
import json
import logging

from core.utils import setup_logger, config
from services.search.search_engine import IntelligentSearchEngine
from services.learning.learning_agent import LinkLearningAgent

logger = setup_logger('search_api', 'logs/search_api.log')

app = FastAPI(
    title="Document Intelligence Search API",
    version="1.0.0",
    description="DSGVO-konforme Dokumentensuche mit KI-Unterstützung"
)

# CORS für WebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globale Instanzen
search_engine = None
learning_agent = None

# Pydantic Models
class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict] = None
    limit: int = 20
    use_ai: bool = True

class SearchResult(BaseModel):
    doc_id: str
    title: str
    score: float
    snippet: str
    categories: List[str]
    keywords: List[str]
    file_path: Optional[str] = None

class DocumentDetail(BaseModel):
    doc_id: str
    title: str
    content: str
    metadata: Dict
    references: List[str]
    semantic_links: Dict[str, float]

class LinkRequest(BaseModel):
    doc1_id: str
    doc2_id: str
    bidirectional: bool = True

@app.on_event("startup")
async def startup_event():
    """Initialisiere Services beim Start"""
    global search_engine, learning_agent
    
    logger.info("Starte Search API...")
    search_engine = IntelligentSearchEngine(str(config.index_path))
    learning_agent = LinkLearningAgent()
    
    # Lade existierende Indizes
    await search_engine.load_indices_async()
    logger.info("Search API bereit")

@app.get("/")
async def root():
    """API Status"""
    return {
        "service": "Document Intelligence Search API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "/search": "Dokumentensuche",
            "/document/{doc_id}": "Dokument-Details",
            "/suggest/{doc_id}": "Ähnliche Dokumente",
            "/link": "Manuelle Verknüpfung",
            "/stats": "System-Statistiken"
        }
    }

@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """Intelligente Dokumentensuche"""
    try:
        logger.info(f"Suche nach: {request.query}")
        
        # Führe Suche durch
        results = await search_engine.search_async(
            request.query,
            filters=request.filters,
            limit=request.limit
        )
        
        # Konvertiere zu Response-Format
        search_results = []
        for doc, score in results[:request.limit]:
            # Erstelle Snippet
            snippet = ""
            for section in doc.sections[:2]:
                snippet += section.get('content', '')[:200] + "... "
            
            search_results.append(SearchResult(
                doc_id=doc.doc_id,
                title=doc.title,
                score=float(score),
                snippet=snippet.strip(),
                categories=doc.categories,
                keywords=doc.keywords[:10],
                file_path=doc.metadata.get('path')
            ))
        
        # Learning: Speichere Suchanfrage
        if results:
            clicked_ids = [r[0].doc_id for r in results[:5]]
            await learning_agent.learn_from_search_async(request.query, clicked_ids)
        
        return search_results
        
    except Exception as e:
        logger.error(f"Suchfehler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: str):
    """Hole Dokument-Details"""
    try:
        doc = await search_engine.get_document_async(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
        
        # Vollständiger Inhalt
        full_content = "\n\n".join([s.get('content', '') for s in doc.sections])
        
        return DocumentDetail(
            doc_id=doc.doc_id,
            title=doc.title,
            content=full_content,
            metadata=doc.metadata,
            references=doc.references,
            semantic_links=doc.semantic_links
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Abrufen von Dokument {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/suggest/{doc_id}", response_model=List[SearchResult])
async def suggest_similar(doc_id: str, limit: int = 10):
    """Schlage ähnliche Dokumente vor"""
    try:
        # Hole Vorschläge vom Learning Agent
        suggestions = learning_agent.suggest_links(doc_id)
        
        # Erweitere mit semantischen Links
        doc = await search_engine.get_document_async(doc_id)
        if doc:
            for link_id, score in doc.semantic_links.items():
                if link_id not in suggestions:
                    suggestions[link_id] = score
        
        # Sortiere und limitiere
        sorted_suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Konvertiere zu SearchResults
        results = []
        for suggested_id, score in sorted_suggestions:
            suggested_doc = await search_engine.get_document_async(suggested_id)
            if suggested_doc:
                snippet = suggested_doc.sections[0].get('content', '')[:200] if suggested_doc.sections else ""
                
                results.append(SearchResult(
                    doc_id=suggested_doc.doc_id,
                    title=suggested_doc.title,
                    score=float(score),
                    snippet=snippet + "...",
                    categories=suggested_doc.categories,
                    keywords=suggested_doc.keywords[:10]
                ))
        
        return results
        
    except Exception as e:
        logger.error(f"Fehler bei Vorschlägen für {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/link")
async def create_manual_link(request: LinkRequest):
    """Erstelle manuelle Verknüpfung zwischen Dokumenten"""
    try:
        learning_agent.add_manual_link(
            request.doc1_id,
            request.doc2_id,
            request.bidirectional
        )
        
        # Aktualisiere auch die Indizes
        await search_engine.update_semantic_link_async(
            request.doc1_id,
            request.doc2_id,
            1.0  # Maximale Ähnlichkeit für manuelle Links
        )
        
        if request.bidirectional:
            await search_engine.update_semantic_link_async(
                request.doc2_id,
                request.doc1_id,
                1.0
            )
        
        return {"status": "success", "message": "Verknüpfung erstellt"}
        
    except Exception as e:
        logger.error(f"Fehler beim Erstellen der Verknüpfung: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics():
    """System-Statistiken"""
    try:
        total_docs = len(search_engine.indices)
        
        # Berechne weitere Stats
        total_keywords = sum(len(doc.keywords) for doc in search_engine.indices.values())
        total_links = sum(len(doc.references) for doc in search_engine.indices.values())
        
        categories = {}
        for doc in search_engine.indices.values():
            for cat in doc.categories:
                categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_documents": total_docs,
            "total_keywords": total_keywords,
            "total_links": total_links,
            "categories": categories,
            "index_path": str(config.index_path),
            "indices_loaded": total_docs > 0
        }
        
    except Exception as e:
        logger.error(f"Fehler bei Statistiken: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
