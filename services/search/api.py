# services/search/api.py - KOMPLETTE DATEI ERSETZEN

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import asyncpg
import json
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

def safe_json_parse(json_str: Any) -> Dict:
    """Safely parse JSON string to dict with fallbacks"""
    if json_str is None:
        return {}
    
    if isinstance(json_str, dict):
        return json_str
    
    if isinstance(json_str, str):
        if json_str.strip() == '' or json_str.strip() == '{}':
            return {}
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Failed to parse JSON: {json_str}")
            return {}
    
    # Fallback for any other type
    return {}

@app.on_event("startup")
async def startup_event():
    global db_pool, qdrant_client
    
    # Database connection with retries
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL environment variable is not set")
        raise RuntimeError("DATABASE_URL environment variable is not set")
    
    logger.info(f"Connecting to database with URL: {db_url}")
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            db_pool = await asyncpg.create_pool(db_url)
            logger.info("Database connection established")
            break
        except Exception as e:
            logger.warning(f"Database connection attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
            else:
                logger.error("Failed to connect to database after multiple attempts")
                raise
    
    # Qdrant client
    qdrant_url = os.getenv('QDRANT_URL', 'http://qdrant:6333')
    logger.info(f"Connecting to Qdrant at: {qdrant_url}")
    qdrant_client = QdrantClient(url=qdrant_url)
    
    try:
        qdrant_client.create_collection(
            collection_name="documents",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        logger.info("Created Qdrant collection 'documents'")
    except Exception as e:
        logger.info(f"Qdrant collection already exists or error: {str(e)}")
    
    logger.info("Search API V2 started successfully")

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
            "/stats": "System statistics",
            "/v1/chat/completions": "OpenAI-compatible chat",
            "/v1/models": "Available models"
        }
    }

async def execute_search(request: SearchRequest) -> List[SearchResult]:
    try:
        # Get embedding from Ollama (with fallback)
        embedding = await get_embedding(request.query)
        
        async with db_pool.acquire() as conn:
            # Log search
            query_id = await conn.fetchval("""
                INSERT INTO search_history (query, query_embedding, results)
                VALUES ($1, $2, $3::jsonb) RETURNING id
            """, request.query, embedding, json.dumps([]))
            
            # Text search in chunks with ROBUST metadata handling
            chunks = await conn.fetch("""
                SELECT 
                    c.id,
                    c.content,
                    c.content_type,
                    c.page_number,
                    c.metadata,
                    d.original_name
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.content ILIKE $1
                AND d.status = 'completed'
                ORDER BY c.created_at DESC
                LIMIT $2
            """, f'%{request.query}%', request.limit)
            
            results = []
            for chunk in chunks:
                try:
                    # ROBUST metadata parsing
                    chunk_metadata = safe_json_parse(chunk['metadata'])
                    
                    # Create SearchResult with safe data
                    result = SearchResult(
                        chunk_id=str(chunk['id']),
                        document_name=chunk['original_name'] or 'Unknown',
                        content=chunk['content'][:500] + '...' if len(chunk['content']) > 500 else chunk['content'],
                        score=0.8,  # Static score for now
                        page=chunk['page_number'],
                        metadata=chunk_metadata
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk['id']}: {e}")
                    # Continue with other results instead of failing completely
                    continue
            
            # Update search results for learning
            if results:
                try:
                    await conn.execute("""
                        UPDATE search_history 
                        SET results = $2::jsonb
                        WHERE id = $1
                    """, query_id, json.dumps([{"chunk_id": r.chunk_id, "score": r.score} for r in results]))
                except Exception as e:
                    logger.warning(f"Failed to update search history: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        # Return empty results instead of crashing
        return []

@app.post("/search", response_model=List[SearchResult])
async def search_endpoint(request: SearchRequest):
    try:
        results = await execute_search(request)
        return results
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        # Return empty results instead of HTTP 500
        return []

@app.post("/upload")
async def upload_document(doc: DocumentUpload):
    async with db_pool.acquire() as conn:
        try:
            # Create source if not exists
            source_id = await conn.fetchval("""
                INSERT INTO sources (name, type, config)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (name, type) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
            """, f"api_{doc.source}", "api", json.dumps({}))
            
            # Create document
            doc_id = await conn.fetchval("""
                INSERT INTO documents (source_id, original_name, file_type, metadata)
                VALUES ($1, $2, $3, $4::jsonb)
                RETURNING id
            """, source_id, doc.name, "text", json.dumps(doc.metadata or {}))
            
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
        try:
            await conn.execute("""
                INSERT INTO feedback_log (chunk_id, query_id, feedback_type, feedback_data)
                VALUES ($1::uuid, $2::uuid, $3, $4::jsonb)
            """, feedback.chunk_id, feedback.query_id, feedback.feedback_type, 
                json.dumps(feedback.feedback_data or {}))
        except Exception as e:
            logger.warning(f"Feedback error: {e}")
    
    return {"status": "feedback recorded"}

@app.get("/stats")
async def get_statistics():
    try:
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
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "embedded_chunks": 0,
            "total_searches": 0,
            "total_feedback": 0,
            "error": str(e)
        }

async def get_embedding(text: str):
    """Get embedding with fallback"""
    try:
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ollama_base_url}/api/embeddings",
                json={"model": default_model, "prompt": text},
                timeout=10.0
            )
            
            if response.status_code == 200:
                return response.json()["embedding"]
    except Exception as e:
        logger.warning(f"Embedding error: {e}")
    
    # Fallback to simple embedding
    import hashlib
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return [float((hash_val >> i) & 1) for i in range(384)]

# OpenAI-compatible endpoints
@app.post("/v1/chat/completions")
async def chat_completions(request: Dict):
    """OpenAI-compatible chat API"""
    try:
        messages = request.get("messages", [])
        user_message = messages[-1].get("content", "") if messages else ""
        
        # Document Search
        search_request = SearchRequest(query=user_message, limit=3)
        search_results = await execute_search(search_request)
        
        # Format response
        content = f"📚 Dokumentensuche für '{user_message}':\n\n"
        if search_results:
            for i, doc in enumerate(search_results, 1):
                content += f"{i}. **{doc.document_name}**\n{doc.content[:150]}...\n\n"
        else:
            content = "🔍 Keine Dokumente gefunden."
        
        return {
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "model": "document-search",
            "object": "chat.completion"
        }
    except Exception as e:
        return {
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "content": f"⚠️ Fehler bei der Dokumentensuche: {str(e)}"
                },
                "finish_reason": "stop"
            }]
        }

@app.get("/v1/models")
async def get_models():
    """OpenAI-compatible models for OpenWebUI"""
    return {
        "data": [
            {
                "id": "document-search",
                "object": "model",
                "owned_by": "system",
                "created": 1234567890
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)