-- Document Intelligence System - PostgreSQL Schema
-- Vollständiges System mit Learning & Feedback Integration
-- Safe initialization with fallbacks

-- =========================================
-- === EXTENSIONS & BASIC SETUP ===
-- =========================================

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

-- Check if vector type exists
DO $$ 
BEGIN
    CREATE DOMAIN vector_type AS FLOAT[];
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END $$;

-- =========================================
-- === SOURCE TRACKING ===
-- =========================================

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

-- =========================================
-- === DOCUMENT MANAGEMENT ===
-- =========================================

-- Master document table with hash-based duplicate prevention
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES sources(id) ON DELETE SET NULL,
    original_name VARCHAR(500) NOT NULL,
    file_type VARCHAR(50),
    file_size BIGINT,
    file_hash VARCHAR(64), -- SHA256 hash for duplicate detection
    metadata JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    deleted_at TIMESTAMP
);

-- Document access tracking for analytics and learning
CREATE TABLE IF NOT EXISTS document_access_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    access_type VARCHAR(50),              -- 'search_result', 'direct_download', 'view'
    access_context TEXT,                  -- Suchbegriff der zum Zugriff führte
    user_session VARCHAR(255),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- =========================================
-- === CHUNK MANAGEMENT ===
-- =========================================

-- Basic document chunks
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(50) DEFAULT 'text',
    page_number INTEGER,
    position JSONB,
    status VARCHAR(50) DEFAULT 'raw',
    embedding FLOAT[], -- Fallback if vector not available
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    enhanced_at TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- Enhanced chunks with comprehensive metadata and source tracking
CREATE TABLE IF NOT EXISTS enhanced_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    original_chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    enhanced_content TEXT NOT NULL,
    original_content TEXT,
    categories JSONB DEFAULT '[]',
    extracted_metadata JSONB DEFAULT '{}',
    detected_references JSONB DEFAULT '[]',
    key_topics JSONB DEFAULT '[]',
    content_type VARCHAR(50) DEFAULT 'text',
    page_number INTEGER,
    -- Source information
    source_file_path TEXT,
    source_drive_link TEXT,
    source_repository TEXT,
    -- Quality metrics
    enhancement_model VARCHAR(50),
    confidence_score FLOAT DEFAULT 0.5,
    processing_time INTEGER,
    quality_score FLOAT DEFAULT 0.5,
    manual_review_needed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =========================================
-- === PROCESSING QUEUE ===
-- =========================================

CREATE TABLE IF NOT EXISTS processing_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    task_type VARCHAR(50) NOT NULL,
    priority INTEGER DEFAULT 5,
    deferred BOOLEAN DEFAULT FALSE,
    retry_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- =========================================
-- === SEARCH & LEARNING SYSTEM ===
-- =========================================

-- Basic search history
CREATE TABLE IF NOT EXISTS search_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    query_embedding FLOAT[],
    results JSONB DEFAULT '[]',
    user_feedback INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Enhanced search history with learning context
CREATE TABLE IF NOT EXISTS enhanced_search_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_query TEXT NOT NULL,
    expanded_query TEXT,                  -- Mit Synonymen erweiterte Query
    search_strategy VARCHAR(50),          -- 'exact', 'semantic', 'hybrid', 'learned'
    results_count INTEGER,
    best_match_score FLOAT,
    user_satisfaction INTEGER,           -- 1-5 Sterne, nachträglich via Feedback
    learning_applied BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Query performance tracking for analytics
CREATE TABLE IF NOT EXISTS query_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    query_embedding FLOAT[],
    results_count INTEGER,
    avg_relevance_score FLOAT,
    user_clicked_result BOOLEAN DEFAULT FALSE,
    click_position INTEGER,                    -- Welches Ergebnis wurde angeklickt?
    search_duration_ms INTEGER,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Synonym learning (automatisch aus Suchanfragen)
CREATE TABLE IF NOT EXISTS learned_synonyms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    term_a VARCHAR(255) NOT NULL,
    term_b VARCHAR(255) NOT NULL,
    confidence_score FLOAT DEFAULT 0.5,
    learning_source VARCHAR(50),         -- 'user_feedback', 'co_occurrence', 'manual'
    verified BOOLEAN DEFAULT FALSE,      -- Manuelle Verifikation durch Admin
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(term_a, term_b)
);

-- =========================================
-- === FEEDBACK SYSTEM ===
-- =========================================

-- Basic feedback log
CREATE TABLE IF NOT EXISTS feedback_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    query_id UUID REFERENCES search_history(id) ON DELETE CASCADE,
    feedback_type VARCHAR(50),
    feedback_data JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Enhanced user feedback for learning
CREATE TABLE IF NOT EXISTS search_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    user_query TEXT NOT NULL,
    feedback_type VARCHAR(50) NOT NULL, -- 'helpful', 'not_helpful', 'wrong_source', 'perfect', 'irrelevant'
    feedback_score INTEGER NOT NULL,    -- -1 (schlecht), 0 (neutral), 1 (gut)
    relevance_rating INTEGER,           -- 1-5 Sterne
    user_comment TEXT,
    user_session VARCHAR(255),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- =========================================
-- === PERFORMANCE INDEXES ===
-- =========================================

-- Document indexes
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_file_hash ON documents (file_hash) WHERE file_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_hash_status ON documents (file_hash, status) WHERE file_hash IS NOT NULL;

-- Document access indexes
CREATE INDEX IF NOT EXISTS idx_document_access_document ON document_access_stats(document_id);
CREATE INDEX IF NOT EXISTS idx_document_access_timestamp ON document_access_stats(timestamp DESC);

-- Chunk indexes
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);
CREATE INDEX IF NOT EXISTS idx_chunks_content_type ON chunks(content_type);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_number);

-- Enhanced chunks indexes
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_document ON enhanced_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_quality ON enhanced_chunks(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_confidence ON enhanced_chunks(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_content_search ON enhanced_chunks USING gin(to_tsvector('german', enhanced_content));

-- Processing queue indexes
CREATE INDEX IF NOT EXISTS idx_processing_queue_status ON processing_queue(status, deferred);
CREATE INDEX IF NOT EXISTS idx_processing_queue_priority ON processing_queue(priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_processing_queue_document ON processing_queue(document_id);

-- Search & learning indexes
CREATE INDEX IF NOT EXISTS idx_search_history_created ON search_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_enhanced_search_history_query ON enhanced_search_history(original_query);
CREATE INDEX IF NOT EXISTS idx_enhanced_search_history_timestamp ON enhanced_search_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_query_analytics_timestamp ON query_analytics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_query_analytics_query ON query_analytics(query_text);
CREATE INDEX IF NOT EXISTS idx_learned_synonyms_terms ON learned_synonyms(term_a, term_b);
CREATE INDEX IF NOT EXISTS idx_learned_synonyms_verified ON learned_synonyms(verified);

-- Feedback indexes
CREATE INDEX IF NOT EXISTS idx_feedback_log_chunk ON feedback_log(chunk_id);
CREATE INDEX IF NOT EXISTS idx_search_feedback_query ON search_feedback(user_query);
CREATE INDEX IF NOT EXISTS idx_search_feedback_timestamp ON search_feedback(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_search_feedback_score ON search_feedback(feedback_score);

-- =========================================
-- === FUNCTIONS & TRIGGERS ===
-- =========================================

-- Cleanup function for chunk management
CREATE OR REPLACE FUNCTION cleanup_old_chunks() RETURNS trigger AS $$
BEGIN
    IF NEW.status = 'enhanced' AND OLD.status = 'precleaned' THEN
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

-- Learning functions
CREATE OR REPLACE FUNCTION learn_synonyms_from_cooccurrence()
RETURNS void AS $$
BEGIN
    -- Finde Begriffe die oft zusammen gesucht werden
    INSERT INTO learned_synonyms (term_a, term_b, confidence_score, learning_source)
    SELECT DISTINCT
        t1.term,
        t2.term,
        0.4, -- Niedrigere Confidence für automatisch gelernte Synonyme
        'co_occurrence'
    FROM (
        SELECT unnest(string_to_array(lower(original_query), ' ')) as term
        FROM enhanced_search_history
        WHERE results_count > 0
    ) t1
    CROSS JOIN (
        SELECT unnest(string_to_array(lower(original_query), ' ')) as term
        FROM enhanced_search_history
        WHERE results_count > 0
    ) t2
    WHERE t1.term != t2.term
      AND length(t1.term) > 3
      AND length(t2.term) > 3
      AND t1.term NOT IN ('und', 'oder', 'der', 'die', 'das', 'ist', 'für', 'von', 'mit', 'über')
      AND t2.term NOT IN ('und', 'oder', 'der', 'die', 'das', 'ist', 'für', 'von', 'mit', 'über')
    ON CONFLICT (term_a, term_b) DO UPDATE 
    SET confidence_score = LEAST(0.8, learned_synonyms.confidence_score + 0.1);
END;
$$ LANGUAGE plpgsql;

-- Update quality scores based on user feedback
CREATE OR REPLACE FUNCTION update_quality_from_feedback()
RETURNS void AS $$
BEGIN
    -- Update enhanced_chunks quality basierend auf durchschnittlichem User-Feedback
    UPDATE enhanced_chunks
    SET quality_score = GREATEST(0.1, LEAST(1.0,
        COALESCE((
            SELECT AVG(feedback_score::float) * 0.3 + quality_score
            FROM search_feedback sf
            WHERE sf.chunk_id = enhanced_chunks.id
            AND sf.timestamp > NOW() - INTERVAL '30 days'
        ), quality_score)
    ))
    WHERE id IN (
        SELECT DISTINCT chunk_id 
        FROM search_feedback 
        WHERE timestamp > NOW() - INTERVAL '1 day'
    );
END;
$$ LANGUAGE plpgsql;

-- Trigger function for automatic quality update
CREATE OR REPLACE FUNCTION trigger_quality_update()
RETURNS trigger AS $$
BEGIN
    -- Update Quality Score des betroffenen Chunks
    UPDATE enhanced_chunks
    SET quality_score = GREATEST(0.1, LEAST(1.0,
        quality_score + (NEW.feedback_score::float * 0.05)
    ))
    WHERE id = NEW.chunk_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
DROP TRIGGER IF EXISTS trigger_cleanup_chunks ON chunks;
CREATE TRIGGER trigger_cleanup_chunks
AFTER UPDATE ON chunks
FOR EACH ROW
EXECUTE FUNCTION cleanup_old_chunks();

DROP TRIGGER IF EXISTS auto_quality_update ON search_feedback;
CREATE TRIGGER auto_quality_update
    AFTER INSERT ON search_feedback
    FOR EACH ROW
    EXECUTE FUNCTION trigger_quality_update();

-- =========================================
-- === ANALYTICS VIEWS ===
-- =========================================

-- Enhanced document stats with duplicate detection and quality metrics
CREATE OR REPLACE VIEW document_stats AS
SELECT 
    d.id,
    d.original_name,
    d.file_type,
    d.file_size,
    CASE 
        WHEN d.file_hash IS NOT NULL THEN LEFT(d.file_hash, 8) || '...'
        ELSE 'no_hash'
    END as hash_preview,
    d.status,
    d.created_at,
    COUNT(DISTINCT c.id) as chunk_count,
    COUNT(DISTINCT CASE WHEN c.status = 'enhanced' THEN c.id END) as enhanced_chunks,
    COUNT(DISTINCT CASE WHEN c.embedding IS NOT NULL THEN c.id END) as embedded_chunks,
    -- Enhanced chunks quality metrics
    COUNT(DISTINCT ec.id) as enhanced_chunks_count,
    AVG(ec.quality_score) as avg_quality_score,
    AVG(ec.confidence_score) as avg_confidence_score,
    -- Access statistics
    COUNT(DISTINCT das.id) as access_count,
    MAX(das.timestamp) as last_accessed,
    -- Check for potential duplicates
    (SELECT COUNT(*) - 1 FROM documents d2 WHERE d2.file_hash = d.file_hash AND d2.file_hash IS NOT NULL) as duplicate_count
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
LEFT JOIN enhanced_chunks ec ON d.id = ec.document_id
LEFT JOIN document_access_stats das ON d.id = das.document_id
WHERE d.status != 'deleted'
GROUP BY d.id, d.original_name, d.file_type, d.file_size, d.file_hash, d.status, d.created_at;

-- Learning insights dashboard
CREATE OR REPLACE VIEW learning_insights AS
SELECT 
    'popular_queries' as insight_type,
    COUNT(*) as count,
    original_query as detail,
    AVG(results_count) as avg_results
FROM enhanced_search_history
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY original_query
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC
LIMIT 10

UNION ALL

SELECT 
    'popular_documents' as insight_type,
    COUNT(*) as count,
    d.original_name as detail,
    AVG(sf.feedback_score::float) as avg_feedback
FROM document_access_stats das
JOIN documents d ON das.document_id = d.id
LEFT JOIN search_feedback sf ON sf.document_id = d.id
WHERE das.timestamp > NOW() - INTERVAL '7 days'
GROUP BY d.id, d.original_name
ORDER BY COUNT(*) DESC
LIMIT 5

UNION ALL

SELECT 
    'learning_progress' as insight_type,
    COUNT(*) as count,
    CASE 
        WHEN verified THEN 'verified_synonyms'
        ELSE 'pending_synonyms'
    END as detail,
    AVG(confidence_score) as avg_confidence
FROM learned_synonyms
GROUP BY verified

UNION ALL

SELECT 
    'user_satisfaction' as insight_type,
    COUNT(*) as count,
    feedback_type as detail,
    AVG(feedback_score::float) as avg_score
FROM search_feedback
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY feedback_type;

-- Document quality overview
CREATE OR REPLACE VIEW document_quality_stats AS
SELECT 
    d.id,
    d.original_name,
    COUNT(ec.id) as enhanced_chunks_count,
    AVG(ec.quality_score) as avg_quality_score,
    AVG(ec.confidence_score) as avg_confidence_score,
    COUNT(sf.id) as feedback_count,
    AVG(sf.feedback_score::float) as avg_user_rating,
    MAX(das.timestamp) as last_accessed
FROM documents d
LEFT JOIN enhanced_chunks ec ON d.id = ec.document_id
LEFT JOIN search_feedback sf ON d.id = sf.document_id
LEFT JOIN document_access_stats das ON d.id = das.document_id
WHERE d.status != 'deleted'
GROUP BY d.id, d.original_name
ORDER BY avg_quality_score DESC NULLS LAST;

-- Search performance monitoring
CREATE OR REPLACE VIEW search_performance_stats AS
SELECT 
    DATE_TRUNC('day', timestamp) as search_date,
    COUNT(*) as total_searches,
    AVG(results_count) as avg_results_per_search,
    AVG(search_duration_ms) as avg_search_time_ms,
    COUNT(CASE WHEN user_clicked_result THEN 1 END) as clicks,
    ROUND(COUNT(CASE WHEN user_clicked_result THEN 1 END)::float / COUNT(*) * 100, 2) as click_through_rate
FROM query_analytics
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', timestamp)
ORDER BY search_date DESC;

-- Processing queue monitoring
CREATE OR REPLACE VIEW queue_stats AS
SELECT 
    status,
    task_type,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (COALESCE(completed_at, NOW()) - created_at))) as avg_processing_time_seconds,
    MIN(created_at) as oldest_task,
    MAX(created_at) as newest_task
FROM processing_queue
GROUP BY status, task_type
ORDER BY status, task_type;

-- =========================================
-- === INITIAL DATA & PERMISSIONS ===
-- =========================================

-- Example synonyms for testing
INSERT INTO learned_synonyms (term_a, term_b, confidence_score, learning_source, verified) VALUES
('roses', 'wood''s', 0.9, 'manual', true),
('metall', 'legierung', 0.8, 'manual', true),
('bismut', 'wismut', 0.95, 'manual', true),
('schmelzpunkt', 'schmelztemperatur', 0.9, 'manual', true),
('zusammensetzung', 'komposition', 0.85, 'manual', true)
ON CONFLICT (term_a, term_b) DO NOTHING;

-- Grant permissions
DO $$ 
BEGIN
    -- Grant permissions only if user exists
    IF EXISTS (SELECT 1 FROM pg_user WHERE usename = current_user) THEN
        GRANT ALL ON ALL TABLES IN SCHEMA public TO current_user;
        GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO current_user;
        GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO current_user;
    END IF;
END $$;

-- =========================================
-- === INITIALIZATION LOG ===
-- =========================================

DO $$ 
BEGIN
    RAISE NOTICE '=== DOCUMENT INTELLIGENCE SYSTEM INITIALIZED ===';
    RAISE NOTICE 'Core Features:';
    RAISE NOTICE '  - Hash-based duplicate detection: ENABLED';
    RAISE NOTICE '  - Enhanced chunks with quality metrics: ENABLED';
    RAISE NOTICE '  - Learning & feedback system: ENABLED';
    RAISE NOTICE '  - Search analytics: ENABLED';
    RAISE NOTICE '  - Synonym learning: ENABLED';
    RAISE NOTICE '';
    RAISE NOTICE 'Database Statistics:';
    RAISE NOTICE '  - Total tables: %', (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public');
    RAISE NOTICE '  - Total indexes: %', (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public');
    RAISE NOTICE '  - Initial synonyms: %', (SELECT COUNT(*) FROM learned_synonyms WHERE verified = true);
    RAISE NOTICE '';
    RAISE NOTICE 'Key Views Available:';
    RAISE NOTICE '  - document_stats (enhanced with quality metrics)';
    RAISE NOTICE '  - learning_insights (learning analytics dashboard)';
    RAISE NOTICE '  - document_quality_stats (quality overview)';
    RAISE NOTICE '  - search_performance_stats (search analytics)';
    RAISE NOTICE '  - queue_stats (processing monitoring)';
    RAISE NOTICE '=============================================';
END $$;