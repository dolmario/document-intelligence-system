from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import asyncpg
import logging
import os
from datetime import datetime
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger('search_api')

app = FastAPI(
    title="Document Intelligence Search API V2",
    version="2.0.0",
    description="Semantic document search with learning"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connections
db_pool = None
qdrant_client = None
ollama_base_url = os.getenv('OLLAMA_URL', 'http://ollama:11434')
default_model = os.getenv('DEFAULT_MODEL', 'mistral')

# Pydantic Models
class SearchRequest(BaseModel):
    query: str
    limit: int = 20
    filters: Optional[Dict] = None

class SearchResult(BaseModel):
    chunk_id: str
    document_name: str
    content: str
    score: float
    page: Optional[int] = None
    metadata: Optional[Dict] = None

class DocumentUpload(BaseModel):
    name: str
    content: str
    source: str = "api"
    metadata: Optional[Dict] = None

class FeedbackRequest(BaseModel):
    query_id: str
    chunk_id: str
    feedback_type: str  # relevant, irrelevant, correction
    feedback_data: Optional[Dict] = None

@app.on_event("startup")
async def startup_event():
    global db_pool, qdrant_client
    
    db_url = os.getenv('DATABASE_URL')
    db_pool = await asyncpg.create_pool(db_url)
    
    qdrant_client = QdrantClient(
        url=os.getenv('QDRANT_URL', 'http://qdrant:6333')
    )
    
    # Create collection if not exists
    try:
        qdrant_client.create_collection(
            collection_name="documents",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    except:
        pass  # Collection already exists
    
    logger.info("Search API V2 started")

@app.on_event("shutdown")
async def shutdown_event():
    if db_pool:
        await db_pool.close()

@app.get("/")
async def root():
    return {
        "service": "Document Intelligence Search API V2",
        "status": "running",
        "endpoints": {
            "/search": "Semantic search",
            "/upload": "Document upload",
            "/feedback": "Learning feedback",
            "/stats": "System statistics"
        }
    }

@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    try:
        # Get embedding from Ollama
        embedding = await get_embedding(request.query)
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name="documents",
            query_vector=embedding,
            limit=request.limit
        )
        
        results = []
        async with db_pool.acquire() as conn:
            # Log search
            query_id = await conn.fetchval("""
                INSERT INTO search_history (query, query_embedding)
                VALUES ($1, $2) RETURNING id
            """, request.query, embedding)
            
            for hit in search_results:
                # Get chunk details
                chunk = await conn.fetchrow("""
                    SELECT c.*, d.original_name
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.id = $1::uuid
                """, hit.id)
                
                if chunk:
                    results.append(SearchResult(
                        chunk_id=str(chunk['id']),
                        document_name=chunk['original_name'],
                        content=chunk['content'],
                        score=hit.score,
                        page=chunk['page_number'],
                        metadata=chunk['metadata']
                    ))
            
            # Update search results for learning
            await conn.execute("""
                UPDATE search_history 
                SET results = $2
                WHERE id = $1
            """, query_id, [{"chunk_id": r.chunk_id, "score": r.score} for r in results])
        
        return results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(doc: DocumentUpload):
    async with db_pool.acquire() as conn:
        try:
            # Create source if not exists
            source_id = await conn.fetchval("""
                INSERT INTO sources (name, type, config)
                VALUES ($1, $2, $3)
                ON CONFLICT DO NOTHING
                RETURNING id
            """, f"api_{doc.source}", "api", {})
            
            # Create document
            doc_id = await conn.fetchval("""
                INSERT INTO documents (source_id, original_name, file_type, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, source_id, doc.name, "text", doc.metadata or {})
            
            # Create processing task
            await conn.execute("""
                INSERT INTO processing_queue (document_id, task_type, priority)
                VALUES ($1, 'extract', 5)
            """, doc_id)
            
            return {"document_id": str(doc_id), "status": "queued"}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO feedback_log (chunk_id, query_id, feedback_type, feedback_data)
            VALUES ($1::uuid, $2::uuid, $3, $4)
        """, feedback.chunk_id, feedback.query_id, feedback.feedback_type, 
            feedback.feedback_data or {})
    
    return {"status": "feedback recorded"}

@app.get("/stats")
async def get_statistics():
    async with db_pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT 
                (SELECT COUNT(*) FROM documents WHERE status != 'deleted') as total_documents,
                (SELECT COUNT(*) FROM chunks) as total_chunks,
                (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) as embedded_chunks,
                (SELECT COUNT(*) FROM search_history) as total_searches,
                (SELECT COUNT(*) FROM feedback_log) as total_feedback
        """)
        
        return dict(stats)

async def get_embedding(text: str):
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ollama_base_url}/api/embeddings",
            json={"model": default_model, "prompt": text}
        )
        
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            # Fallback to simple embedding
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return [float((hash_val >> i) & 1) for i in range(384)]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
