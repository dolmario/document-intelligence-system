-- Document Intelligence System - PostgreSQL Schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Source tracking
CREATE TABLE sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL, -- folder, api, email, n8n
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Master document table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES sources(id),
    original_name VARCHAR(500) NOT NULL,
    file_type VARCHAR(50),
    file_size BIGINT,
    file_hash VARCHAR(64) UNIQUE,
    metadata JSONB,
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed, deleted
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    deleted_at TIMESTAMP
);

-- Document chunks
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(50), -- text, table, image_description
    page_number INTEGER,
    position JSONB, -- {x, y, width, height} for spatial reference
    status VARCHAR(50) DEFAULT 'raw', -- raw, precleaned, enhanced, embedded
    embedding vector(384), -- for semantic search
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    enhanced_at TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- Processing queue
CREATE TABLE processing_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id),
    task_type VARCHAR(50), -- ocr, extract, enhance, embed
    priority INTEGER DEFAULT 5,
    deferred BOOLEAN DEFAULT FALSE, -- for night jobs
    retry_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Search history for learning
CREATE TABLE search_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    query_embedding vector(384),
    results JSONB, -- [{chunk_id, score, clicked}]
    user_feedback INTEGER, -- 1-5 rating
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feedback for learning
CREATE TABLE feedback_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID REFERENCES chunks(id),
    query_id UUID REFERENCES search_history(id),
    feedback_type VARCHAR(50), -- relevant, irrelevant, correction
    feedback_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_source ON documents(source_id);
CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_status ON chunks(status);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_processing_queue_status ON processing_queue(status, deferred);
CREATE INDEX idx_search_history_created ON search_history(created_at DESC);

-- Cleanup trigger
CREATE OR REPLACE FUNCTION cleanup_old_chunks() RETURNS trigger AS $$
BEGIN
    IF NEW.status = 'enhanced' AND OLD.status = 'precleaned' THEN
        -- Mark old version for deletion after 24h
        UPDATE chunks 
        SET status = 'archived', 
            metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{archived_at}', to_jsonb(CURRENT_TIMESTAMP))
        WHERE document_id = NEW.document_id 
        AND chunk_index = NEW.chunk_index 
        AND status = 'raw';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_cleanup_chunks
AFTER UPDATE ON chunks
FOR EACH ROW
EXECUTE FUNCTION cleanup_old_chunks();

-- Stats view
CREATE VIEW document_stats AS
SELECT 
    d.id,
    d.original_name,
    d.status,
    COUNT(DISTINCT c.id) as chunk_count,
    COUNT(DISTINCT CASE WHEN c.status = 'enhanced' THEN c.id END) as enhanced_chunks,
    COUNT(DISTINCT CASE WHEN c.embedding IS NOT NULL THEN c.id END) as embedded_chunks
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
GROUP BY d.id;
