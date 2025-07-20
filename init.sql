-- Semantic Document Finder System - PostgreSQL Schema
-- Safe initialization with fallbacks

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Vector extension (optional - for embeddings)
DO $$ 
BEGIN
    CREATE EXTENSION IF NOT EXISTS "vector";
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'vector extension not available - embeddings will use FLOAT[]';
END $$;

-- Source tracking
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, type)
);

-- Insert default sources
INSERT INTO sources (name, type) VALUES ('manual_upload', 'api') ON CONFLICT DO NOTHING;
INSERT INTO sources (name, type) VALUES ('n8n_upload', 'n8n') ON CONFLICT DO NOTHING;

-- Master document table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES sources(id) ON DELETE SET NULL,
    original_name VARCHAR(500) NOT NULL,
    file_type VARCHAR(50),
    file_size BIGINT,
    file_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    deleted_at TIMESTAMP
);

-- Raw document chunks (OCR output)
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(50) DEFAULT 'text',
    page_number INTEGER,
    position JSONB,
    status VARCHAR(50) DEFAULT 'raw',
    embedding FLOAT[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- Enhanced chunks (LLM processed)
CREATE TABLE IF NOT EXISTS enhanced_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    original_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    chunk_index INTEGER NOT NULL,
    
    -- Enhanced Content (LLM processed)
    enhanced_content TEXT NOT NULL,
    original_content TEXT,
    
    -- LLM Extracted Data
    categories JSONB DEFAULT '[]'::jsonb,
    extracted_metadata JSONB DEFAULT '{}'::jsonb,
    detected_references JSONB DEFAULT '[]'::jsonb,
    key_topics JSONB DEFAULT '[]'::jsonb,
    
    -- Content Structure
    content_type VARCHAR(50) DEFAULT 'text',
    content_structure JSONB DEFAULT '{}'::jsonb,
    page_number INTEGER,
    position JSONB,
    
    -- LLM Processing Info
    enhancement_model VARCHAR(50),
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    processing_time INTEGER,
    enhancement_prompt TEXT,
    
    -- Vector Search
    embedding FLOAT[],
    
    -- Quality Control
    quality_score FLOAT DEFAULT 0.5,
    manual_review_needed BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    enhanced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP,
    
    UNIQUE(document_id, chunk_index)
);

-- Table content storage (for structured data)
CREATE TABLE IF NOT EXISTS table_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    enhanced_chunk_id UUID REFERENCES enhanced_chunks(id) ON DELETE CASCADE,
    
    -- Table Data
    table_data JSONB NOT NULL,
    table_headers JSONB,
    table_summary TEXT,
    row_count INTEGER,
    column_count INTEGER,
    
    -- Original Image (fallback when LLM can't read table)
    original_image_base64 TEXT,
    image_ocr_text TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Image content storage (for images/diagrams)
CREATE TABLE IF NOT EXISTS image_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    enhanced_chunk_id UUID REFERENCES enhanced_chunks(id) ON DELETE CASCADE,
    
    -- Image Data
    image_base64 TEXT NOT NULL,
    image_format VARCHAR(10),
    image_dimensions JSONB,
    
    -- OCR & LLM Analysis
    ocr_text TEXT,
    llm_description TEXT,
    detected_elements JSONB,
    
    -- Classification
    image_type VARCHAR(50),
    business_relevance VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing queue
CREATE TABLE IF NOT EXISTS processing_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    task_type VARCHAR(50) NOT NULL,
    priority INTEGER DEFAULT 5,
    deferred BOOLEAN DEFAULT FALSE,
    retry_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    payload JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Search history for learning
CREATE TABLE IF NOT EXISTS search_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    query_embedding FLOAT[],
    results JSONB DEFAULT '[]',
    user_feedback INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feedback for learning (prepare for future)
CREATE TABLE IF NOT EXISTS enhancement_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    enhanced_chunk_id UUID REFERENCES enhanced_chunks(id) ON DELETE CASCADE,
    feedback_type VARCHAR(50),
    user_feedback INTEGER CHECK (user_feedback IN (-1, 0, 1)),
    feedback_comment TEXT,
    search_query TEXT,
    user_session_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing statistics
CREATE TABLE IF NOT EXISTS processing_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Processing Metrics
    total_chunks INTEGER,
    enhanced_chunks INTEGER,
    failed_chunks INTEGER,
    
    -- Timing
    ocr_duration INTEGER,
    enhancement_duration INTEGER,
    total_processing_time INTEGER,
    
    -- Quality Metrics
    avg_confidence_score FLOAT,
    high_quality_chunks INTEGER,
    review_needed_chunks INTEGER,
    
    -- Model Used
    enhancement_model VARCHAR(50),
    batch_size INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- System performance monitoring
CREATE TABLE IF NOT EXISTS system_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(20),
    context JSONB DEFAULT '{}'::jsonb,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_document ON enhanced_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_content_type ON enhanced_chunks(content_type);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_categories ON enhanced_chunks USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_topics ON enhanced_chunks USING GIN(key_topics);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_confidence ON enhanced_chunks(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_quality ON enhanced_chunks(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_processing_queue_status ON processing_queue(status, deferred);
CREATE INDEX IF NOT EXISTS idx_search_history_created ON search_history(created_at DESC);

-- GIN Index für JSONB search
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_metadata_gin ON enhanced_chunks USING GIN(extracted_metadata);

-- Cleanup function (automatically archive raw chunks after enhancement)
CREATE OR REPLACE FUNCTION cleanup_raw_chunks() RETURNS TRIGGER AS $$
BEGIN
    -- When enhanced chunk is created, mark raw chunk as 'archived'
    UPDATE chunks 
    SET status = 'archived',
        metadata = jsonb_set(
            COALESCE(metadata, '{}'::jsonb), 
            '{archived_at}', 
            to_jsonb(CURRENT_TIMESTAMP),
            true
        )
    WHERE id = NEW.original_chunk_id
    AND status != 'archived';
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic cleanup
DROP TRIGGER IF EXISTS trigger_cleanup_raw_chunks ON enhanced_chunks;
CREATE TRIGGER trigger_cleanup_raw_chunks
    AFTER INSERT ON enhanced_chunks
    FOR EACH ROW
    EXECUTE FUNCTION cleanup_raw_chunks();

-- Auto-update document status based on processing
CREATE OR REPLACE FUNCTION update_document_status() RETURNS TRIGGER AS $$
BEGIN
    UPDATE documents 
    SET status = CASE 
        WHEN (
            SELECT COUNT(*) FROM enhanced_chunks 
            WHERE document_id = NEW.document_id
        ) >= (
            SELECT COUNT(*) FROM chunks 
            WHERE document_id = NEW.document_id AND status != 'archived'
        ) THEN 'enhanced'
        ELSE 'processing'
    END
    WHERE id = NEW.document_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_document_status ON enhanced_chunks;
CREATE TRIGGER trigger_update_document_status
    AFTER INSERT ON enhanced_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_document_status();

-- Useful views for monitoring
CREATE OR REPLACE VIEW document_processing_status AS
SELECT 
    d.id,
    d.original_name,
    d.status as document_status,
    COUNT(DISTINCT c.id) as raw_chunks,
    COUNT(DISTINCT ec.id) as enhanced_chunks,
    COUNT(DISTINCT tc.id) as table_chunks,
    COUNT(DISTINCT ic.id) as image_chunks,
    AVG(ec.confidence_score) as avg_confidence,
    AVG(ec.quality_score) as avg_quality,
    ps.total_processing_time,
    ps.enhancement_model
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id AND c.status != 'deleted'
LEFT JOIN enhanced_chunks ec ON d.id = ec.document_id
LEFT JOIN table_content tc ON ec.id = tc.enhanced_chunk_id
LEFT JOIN image_content ic ON ec.id = ic.enhanced_chunk_id
LEFT JOIN processing_stats ps ON d.id = ps.document_id
GROUP BY d.id, d.original_name, d.status, ps.total_processing_time, ps.enhancement_model;

-- Enhanced search function (works with both raw and enhanced chunks)
CREATE OR REPLACE FUNCTION search_all_content(
    search_query TEXT,
    limit_results INTEGER DEFAULT 20
) RETURNS TABLE (
    chunk_id UUID,
    document_name TEXT,
    content TEXT,
    chunk_type VARCHAR(20),
    confidence_score FLOAT,
    source VARCHAR(20),
    categories JSONB,
    key_topics JSONB
) AS $$
BEGIN
    RETURN QUERY
    -- Enhanced chunks first (higher priority)
    SELECT 
        ec.id,
        d.original_name,
        ec.enhanced_content,
        'enhanced'::VARCHAR(20),
        ec.confidence_score,
        'llm'::VARCHAR(20),
        ec.categories,
        ec.key_topics
    FROM enhanced_chunks ec
    JOIN documents d ON ec.document_id = d.id
    WHERE LOWER(ec.enhanced_content) LIKE LOWER('%' || search_query || '%')
       OR EXISTS (
           SELECT 1 FROM jsonb_array_elements_text(ec.categories) AS cat
           WHERE LOWER(cat) LIKE LOWER('%' || search_query || '%')
       )
       OR EXISTS (
           SELECT 1 FROM jsonb_array_elements_text(ec.key_topics) AS topic
           WHERE LOWER(topic) LIKE LOWER('%' || search_query || '%')
       )
    
    UNION ALL
    
    -- Raw chunks as fallback (only if no enhanced version exists)
    SELECT 
        c.id,
        d.original_name,
        c.content,
        'raw'::VARCHAR(20),
        0.5::FLOAT,
        'ocr'::VARCHAR(20),
        '[]'::JSONB,
        '[]'::JSONB
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE LOWER(c.content) LIKE LOWER('%' || search_query || '%')
      AND c.status != 'archived'
      AND NOT EXISTS (
          SELECT 1 FROM enhanced_chunks ec 
          WHERE ec.original_chunk_id = c.id
      )
    
    ORDER BY confidence_score DESC, chunk_type DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions for all users
DO $$ 
BEGIN
    -- Grant permissions to current user
    IF EXISTS (SELECT 1 FROM pg_user WHERE usename = current_user) THEN
        GRANT ALL ON ALL TABLES IN SCHEMA public TO current_user;
        GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO current_user;
        GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO current_user;
    END IF;
END $$;;