# services/search/api_v2.py - ENHANCED SEARCH API

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import asyncio
import asyncpg
import json
import logging
import os
from datetime import datetime
from uuid import UUID
import httpx
from contextlib import asynccontextmanager

logger = logging.getLogger('enhanced_search_api')

# === PYDANTIC MODELS ===

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(20, ge=1, le=100)
    filters: Optional[Dict] = None
    search_type: str = Field("hybrid", pattern="^(text|semantic|hybrid)$")
    min_confidence: float = Field(0.3, ge=0.0, le=1.0)
    include_raw_fallback: bool = True

class EnhancedSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    content_type: str
    categories: List[str] = []
    key_topics: List[str] = []
    confidence_score: float
    quality_score: float
    relevance_score: float
    page: Optional[int] = None
    enhancement_model: Optional[str] = None
    metadata: Optional[Dict] = None

class DocumentUpload(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    source: str = "api_v2"
    metadata: Optional[Dict] = None
    priority: int = Field(5, ge=1, le=10)

class FeedbackRequest(BaseModel):
    chunk_id: str
    query: str
    feedback_type: str = Field(..., pattern="^(relevant|irrelevant|quality_good|quality_poor|correction)$")
    feedback_score: int = Field(..., ge=-1, le=1)  # -1: bad, 0: neutral, 1: good
    feedback_comment: Optional[str] = None

class AnalyticsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    enhanced_chunks: int
    total_searches: int
    avg_confidence_score: float
    high_quality_chunks: int
    documents_processing: int
    recent_activity: Dict

# === FASTAPI APP SETUP ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

app = FastAPI(
    title="Semantic Document Finder Search API V2",
    version="2.1.0",
    description="Enhanced semantic document search with LLM processing and learning",
    lifespan=lifespan
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
ollama_base_url = os.getenv('OLLAMA_URL', 'http://ollama:11434')
default_model = os.getenv('DEFAULT_MODEL', 'mistral')

# === UTILITY FUNCTIONS ===

def safe_json_parse(json_str: Any) -> Dict:
    """Safely parse JSON string to dict with fallbacks"""
    if json_str is None:
        return {}
    if isinstance(json_str, dict):
        return json_str
    if isinstance(json_str, str):
        if json_str.strip() in ['', '{}', '[]']:
            return {}
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Failed to parse JSON: {json_str[:100]}...")
            return {}
    return {}

async def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama with fallback"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{ollama_base_url}/api/embeddings",
                json={"model": default_model, "prompt": text[:1000]}  # Limit text length
            )
            if response.status_code == 200:
                return response.json()["embedding"]
    except Exception as e:
        logger.warning(f"Embedding error: {e}")
    
    # Fallback to simple hash-based embedding
    import hashlib
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return [float((hash_val >> i) & 1) for i in range(384)]

# === STARTUP/SHUTDOWN ===

async def startup_event():
    global db_pool
    
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    
    logger.info(f"Connecting to database...")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            db_pool = await asyncpg.create_pool(db_url, min_size=2, max_size=20)
            logger.info("Database connection established")
            break
        except Exception as e:
            logger.warning(f"Database connection attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
            else:
                raise
    
    logger.info("Enhanced Search API V2 started successfully")

async def shutdown_event():
    if db_pool:
        await db_pool.close()

# === SEARCH FUNCTIONS ===

async def search_enhanced_content(request: SearchRequest) -> List[EnhancedSearchResult]:
    """Search in enhanced_chunks with multiple strategies"""
    try:
        async with db_pool.acquire() as conn:
            # Log search for analytics
            search_id = await conn.fetchval("""
                INSERT INTO search_history (query, query_embedding, results)
                VALUES ($1, $2, $3::jsonb) RETURNING id
            """, request.query, await get_embedding(request.query), json.dumps([]))
            
            results = []
            
            # Strategy 1: Enhanced chunks search (primary)
            if request.search_type in ['hybrid', 'text']:
                enhanced_results = await search_enhanced_chunks(conn, request)
                results.extend(enhanced_results)
            
            # Strategy 2: Fallback to raw chunks if needed and allowed
            if request.include_raw_fallback and len(results) < request.limit // 2:
                raw_results = await search_raw_chunks_fallback(conn, request, len(results))
                results.extend(raw_results)
            
            # Update search history with results
            if results:
                result_summary = [{"chunk_id": r.chunk_id, "score": r.relevance_score} for r in results[:10]]
                await conn.execute("""
                    UPDATE search_history 
                    SET results = $2::jsonb
                    WHERE id = $1
                """, search_id, json.dumps(result_summary))
            
            # Sort by relevance and limit
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:request.limit]
            
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

async def search_enhanced_chunks(conn, request: SearchRequest) -> List[EnhancedSearchResult]:
    """Search in enhanced_chunks table"""
    try:
        # Complex search query with relevance scoring
        chunks = await conn.fetch("""
            WITH search_results AS (
                SELECT 
                    ec.id,
                    ec.document_id,
                    ec.enhanced_content,
                    ec.content_type,
                    ec.categories,
                    ec.key_topics,
                    ec.extracted_metadata,
                    ec.confidence_score,
                    ec.quality_score,
                    ec.page_number,
                    ec.enhancement_model,
                    d.original_name,
                    
                    -- Relevance scoring
                    CASE 
                        -- Title/filename match (highest priority)
                        WHEN LOWER(d.original_name) LIKE LOWER($1) THEN 1.0
                        
                        -- Enhanced content exact match
                        WHEN LOWER(ec.enhanced_content) LIKE LOWER($1) THEN 0.9
                        
                        -- Categories match
                        WHEN EXISTS (
                            SELECT 1 FROM jsonb_array_elements_text(ec.categories) AS cat
                            WHERE LOWER(cat) LIKE LOWER($1)
                        ) THEN 0.8
                        
                        -- Key topics match
                        WHEN EXISTS (
                            SELECT 1 FROM jsonb_array_elements_text(ec.key_topics) AS topic
                            WHERE LOWER(topic) LIKE LOWER($1)
                        ) THEN 0.7
                        
                        -- Metadata keywords match
                        WHEN ec.extracted_metadata::text ILIKE $1 THEN 0.6
                        
                        -- Partial content match
                        WHEN LOWER(ec.enhanced_content) LIKE LOWER($2) THEN 0.5
                        
                        ELSE 0.3
                    END as base_relevance,
                    
                    -- Boost factors
                    CASE 
                        WHEN ec.quality_score > 0.8 THEN 0.2
                        WHEN ec.quality_score > 0.6 THEN 0.1
                        ELSE 0.0
                    END as quality_boost,
                    
                    CASE 
                        WHEN ec.confidence_score > 0.8 THEN 0.1
                        WHEN ec.confidence_score > 0.6 THEN 0.05
                        ELSE 0.0
                    END as confidence_boost
                    
                FROM enhanced_chunks ec
                JOIN documents d ON ec.document_id = d.id
                WHERE d.status != 'deleted'
                  AND ec.confidence_score >= $3
                  AND (
                      LOWER(ec.enhanced_content) LIKE LOWER($1)
                      OR LOWER(ec.enhanced_content) LIKE LOWER($2)
                      OR LOWER(d.original_name) LIKE LOWER($1)
                      OR EXISTS (
                          SELECT 1 FROM jsonb_array_elements_text(ec.categories) AS cat
                          WHERE LOWER(cat) LIKE LOWER($1)
                      )
                      OR EXISTS (
                          SELECT 1 FROM jsonb_array_elements_text(ec.key_topics) AS topic
                          WHERE LOWER(topic) LIKE LOWER($1)
                      )
                      OR ec.extracted_metadata::text ILIKE $1
                  )
            )
            SELECT *,
                   (base_relevance + quality_boost + confidence_boost) as final_relevance
            FROM search_results
            ORDER BY final_relevance DESC, confidence_score DESC, quality_score DESC
            LIMIT $4
        """, 
        f'%{request.query}%',  # $1 - exact query pattern
        f'%{" ".join(request.query.split()[:3])}%',  # $2 - partial query (first 3 words)
        request.min_confidence,  # $3
        request.limit  # $4
        )
        
        results = []
        for chunk in chunks:
            try:
                # Parse JSON fields safely
                categories = safe_json_parse(chunk['categories'])
                key_topics = safe_json_parse(chunk['key_topics'])
                metadata = safe_json_parse(chunk['extracted_metadata'])
                
                # Convert to list if needed
                if not isinstance(categories, list):
                    categories = []
                if not isinstance(key_topics, list):
                    key_topics = []
                
                result = EnhancedSearchResult(
                    chunk_id=str(chunk['id']),
                    document_id=str(chunk['document_id']),
                    document_name=chunk['original_name'] or 'Unknown',
                    content=chunk['enhanced_content'][:500] + ('...' if len(chunk['enhanced_content']) > 500 else ''),
                    content_type=chunk['content_type'] or 'text',
                    categories=categories,
                    key_topics=key_topics,
                    confidence_score=float(chunk['confidence_score'] or 0.5),
                    quality_score=float(chunk['quality_score'] or 0.5),
                    relevance_score=float(chunk['final_relevance'] or 0.3),
                    page=chunk['page_number'],
                    enhancement_model=chunk['enhancement_model'],
                    metadata=metadata
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing enhanced chunk {chunk['id']}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        return []

async def search_raw_chunks_fallback(conn, request: SearchRequest, current_count: int) -> List[EnhancedSearchResult]:
    """Fallback search in raw chunks when enhanced chunks are insufficient"""
    try:
        remaining_limit = request.limit - current_count
        if remaining_limit <= 0:
            return []
        
        chunks = await conn.fetch("""
            SELECT 
                c.id,
                c.document_id,
                c.content,
                c.content_type,
                c.page_number,
                c.metadata,
                d.original_name
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.content ILIKE $1
              AND c.status NOT IN ('archived', 'deleted')
              AND d.status != 'deleted'
              AND NOT EXISTS (
                  SELECT 1 FROM enhanced_chunks ec 
                  WHERE ec.original_chunk_id = c.id
              )
            ORDER BY c.created_at DESC
            LIMIT $2
        """, f'%{request.query}%', remaining_limit)
        
        results = []
        for chunk in chunks:
            try:
                metadata = safe_json_parse(chunk['metadata'])
                
                result = EnhancedSearchResult(
                    chunk_id=str(chunk['id']),
                    document_id=str(chunk['document_id']),
                    document_name=chunk['original_name'] or 'Unknown',
                    content=chunk['content'][:500] + ('...' if len(chunk['content']) > 500 else ''),
                    content_type=chunk['content_type'] or 'text',
                    categories=['[Processing]'],  # Indicate it's being processed
                    key_topics=[],
                    confidence_score=0.4,  # Lower confidence for raw chunks
                    quality_score=0.3,
                    relevance_score=0.4,
                    page=chunk['page_number'],
                    enhancement_model='pending',
                    metadata=metadata
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing raw chunk {chunk['id']}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Raw fallback search error: {e}")
        return []

# === API ENDPOINTS ===

@app.get("/")
async def root():
    return {
        "service": "Semantic Document Finder Search API V2",
        "version": "2.1.0",
        "status": "running",
        "features": {
            "enhanced_search": True,
            "llm_processing": True,
            "table_extraction": True,
            "image_processing": True,
            "privacy_protection": True,
            "learning_feedback": True
        },
        "endpoints": {
            "/search": "Enhanced semantic search",
            "/upload": "Document upload with priority",
            "/feedback": "Learning feedback",
            "/analytics": "System analytics",
            "/health": "Health check",
            "/v1/chat/completions": "OpenAI-compatible chat"
        }
    }

@app.post("/search", response_model=List[EnhancedSearchResult])
async def search_endpoint(request: SearchRequest):
    """Enhanced search endpoint with multiple strategies"""
    try:
        results = await search_enhanced_content(request)
        return results
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        return []

@app.post("/upload")
async def upload_document(doc: DocumentUpload, background_tasks: BackgroundTasks):
    """Upload document with enhanced processing queue"""
    async with db_pool.acquire() as conn:
        try:
            # Create source if not exists
            source_id = await conn.fetchval("""
                INSERT INTO sources (name, type, config)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (name, type) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
            """, f"api_v2_{doc.source}", "api", json.dumps({"api_version": "2.1"}))
            
            # Create document
            doc_id = await conn.fetchval("""
                INSERT INTO documents (source_id, original_name, file_type, metadata, enhancement_priority)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                RETURNING id
            """, source_id, doc.name, "text", json.dumps(doc.metadata or {}), doc.priority)
            
            # Create processing task with priority
            queue_id = await conn.fetchval("""
                INSERT INTO processing_queue (document_id, task_type, priority, payload)
                VALUES ($1, 'extract', $2, $3::jsonb)
                RETURNING id
            """, doc_id, doc.priority, json.dumps({
                "api_version": "2.1",
                "content_preview": doc.content[:100],
                "upload_method": "enhanced_api"
            }))
            
            return {
                "document_id": str(doc_id),
                "queue_id": str(queue_id),
                "status": "queued",
                "priority": doc.priority,
                "estimated_processing_time": "2-5 minutes"
            }
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for learning system"""
    async with db_pool.acquire() as conn:
        try:
            # Record feedback
            await conn.execute("""
                INSERT INTO enhancement_feedback (enhanced_chunk_id, feedback_type, user_feedback, feedback_comment, search_query)
                VALUES ($1::uuid, $2, $3, $4, $5)
            """, feedback.chunk_id, feedback.feedback_type, feedback.feedback_score, 
                feedback.feedback_comment, feedback.query)
            
            # Update chunk quality score based on feedback
            await conn.execute("""
                UPDATE enhanced_chunks 
                SET quality_score = GREATEST(0.0, LEAST(1.0, 
                    quality_score + ($2::float * 0.1)
                ))
                WHERE id = $1::uuid
            """, feedback.chunk_id, feedback.feedback_score)
            
            return {"status": "feedback recorded", "impact": "quality score updated"}
            
        except Exception as e:
            logger.warning(f"Feedback error: {e}")
            return {"status": "feedback recorded", "impact": "logged for review"}

@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get comprehensive system analytics"""
    try:
        async with db_pool.acquire() as conn:
            # Basic stats
            stats = await conn.fetchrow("""
                SELECT 
                    (SELECT COUNT(*) FROM documents WHERE status != 'deleted') as total_documents,
                    (SELECT COUNT(*) FROM chunks) as total_chunks,
                    (SELECT COUNT(*) FROM enhanced_chunks) as enhanced_chunks,
                    (SELECT COUNT(*) FROM search_history) as total_searches,
                    (SELECT AVG(confidence_score) FROM enhanced_chunks) as avg_confidence_score,
                    (SELECT COUNT(*) FROM enhanced_chunks WHERE quality_score > 0.8) as high_quality_chunks,
                    (SELECT COUNT(*) FROM documents WHERE status = 'processing') as documents_processing
            """)
            
            # Recent activity
            recent_activity = await conn.fetchrow("""
                SELECT 
                    (SELECT COUNT(*) FROM documents WHERE created_at > NOW() - INTERVAL '24 hours') as documents_today,
                    (SELECT COUNT(*) FROM enhanced_chunks WHERE created_at > NOW() - INTERVAL '24 hours') as chunks_enhanced_today,
                    (SELECT COUNT(*) FROM search_history WHERE created_at > NOW() - INTERVAL '24 hours') as searches_today,
                    (SELECT COUNT(*) FROM enhancement_feedback WHERE created_at > NOW() - INTERVAL '24 hours') as feedback_today
            """)
            
            return AnalyticsResponse(
                total_documents=stats['total_documents'] or 0,
                total_chunks=stats['total_chunks'] or 0,
                enhanced_chunks=stats['enhanced_chunks'] or 0,
                total_searches=stats['total_searches'] or 0,
                avg_confidence_score=float(stats['avg_confidence_score'] or 0.0),
                high_quality_chunks=stats['high_quality_chunks'] or 0,
                documents_processing=stats['documents_processing'] or 0,
                recent_activity={
                    "documents_uploaded_today": recent_activity['documents_today'] or 0,
                    "chunks_enhanced_today": recent_activity['chunks_enhanced_today'] or 0,
                    "searches_today": recent_activity['searches_today'] or 0,
                    "feedback_submitted_today": recent_activity['feedback_today'] or 0
                }
            )
            
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        # Return safe defaults
        return AnalyticsResponse(
            total_documents=0, total_chunks=0, enhanced_chunks=0,
            total_searches=0, avg_confidence_score=0.0, high_quality_chunks=0,
            documents_processing=0, recent_activity={}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            
        # Check Ollama
        ollama_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{ollama_base_url}/api/tags")
                ollama_status = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            ollama_status = "unreachable"
        
        return {
            "status": "healthy",
            "database": "connected",
            "ollama": ollama_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# === OPENAI COMPATIBLE ENDPOINTS ===

@app.post("/v1/chat/completions")
async def chat_completions(request: Dict):
    """OpenAI-compatible chat API with document search"""
    try:
        messages = request.get("messages", [])
        user_message = messages[-1].get("content", "") if messages else ""
        
        # Enhanced document search
        search_request = SearchRequest(
            query=user_message, 
            limit=5, 
            search_type="hybrid",
            min_confidence=0.4
        )
        search_results = await search_enhanced_content(search_request)
        
        # Format response with enhanced data
        if search_results:
            content = f"📚 **Erweiterte Dokumentensuche** für '{user_message}':\n\n"
            for i, doc in enumerate(search_results, 1):
                confidence_emoji = "🟢" if doc.confidence_score > 0.8 else "🟡" if doc.confidence_score > 0.6 else "🔴"
                content += f"{i}. {confidence_emoji} **{doc.document_name}**"
                if doc.categories:
                    content += f" _{', '.join(doc.categories[:2])}_"
                content += f"\n   📄 {doc.content[:150]}...\n"
                if doc.key_topics:
                    content += f"   🏷️ {', '.join(doc.key_topics[:3])}\n"
                content += f"   📊 Qualität: {doc.quality_score:.1%} | Seite: {doc.page or 'N/A'}\n\n"
        else:
            content = f"🔍 Keine Dokumente gefunden für: '{user_message}'\n\n💡 **Tipp**: Versuche spezifischere Suchbegriffe oder prüfe ob deine Dokumente bereits verarbeitet wurden."
        
        return {
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "model": "document-search-enhanced",
            "object": "chat.completion",
            "usage": {
                "total_tokens": len(content.split()),
                "completion_tokens": len(content.split()),
                "prompt_tokens": len(user_message.split())
            }
        }
    except Exception as e:
        return {
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "content": f"⚠️ Fehler bei der erweiterten Dokumentensuche: {str(e)}"
                },
                "finish_reason": "stop"
            }]
        }

@app.get("/v1/models")
async def get_models():
    """OpenAI-compatible models endpoint"""
    return {
        "data": [
            {
                "id": "document-search-enhanced",
                "object": "model",
                "owned_by": "semantic-doc-finder",
                "created": 1234567890,
                "description": "Enhanced document search with LLM processing"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)