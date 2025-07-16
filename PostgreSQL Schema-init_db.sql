-- Document Intelligence System - PostgreSQL Schema
-- Safe initialization with fallbacks and structured setup

-- 1. Erweiterung von PostgreSQL mit UUID und Verschlüsselung
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 2. Vector-Erweiterung (optional - für Embeddings)
DO $$ 
BEGIN
    CREATE EXTENSION IF NOT EXISTS "vector";
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'vector extension not available - embeddings will use FLOAT[]';
END $$;

-- 3. Fallback: Vector-Type als FLOAT[] definieren, falls 'vector' nicht verfügbar
DO $$ 
BEGIN
    CREATE DOMAIN vector_type AS FLOAT[];
EXCEPTION
    WHEN duplicate_object THEN
        NULL;
END $$;

-- 4. Quellen-Tabelle (z. B. Upload per API oder n8n)
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, type)
);

-- Standardwerte für sources
INSERT INTO sources (name, type) VALUES ('manual_upload', 'api') ON CONFLICT DO NOTHING;
INSERT INTO sources (name, type) VALUES ('n8n_upload', 'n8n') ON CONFLICT DO NOTHING;

-- 5. Dokumente-Tabelle (Metadaten zu hochgeladenen Dateien)
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

-- 6. Chunks-Tabelle (aufgeteilte Textabschnitte eines Dokuments)
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(50) DEFAULT 'text',
    page_number INTEGER,
    position JSONB,
    status VARCHAR(50) DEFAULT 'raw',
    embedding FLOAT[], -- Fallback wenn kein vector-type verfügbar
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    enhanced_at TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- 7. Verarbeitungsschlange für asynchrone Aufgaben
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

-- 8. Suchverlauf für Lernsysteme (z. B. Relevanzverbesserung)
CREATE TABLE IF NOT EXISTS search_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    query_embedding FLOAT[],
    results JSONB DEFAULT '[]',
    user_feedback INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 9. Feedback-Tabelle zur Bewertung von Suchtreffern
CREATE TABLE IF NOT EXISTS feedback_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    query_id UUID REFERENCES search_history(id) ON DELETE CASCADE,
    feedback_type VARCHAR(50),
    feedback_data JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 10. Indizes für Performance
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);
CREATE INDEX IF NOT EXISTS idx_processing_queue_status ON processing_queue(status, deferred);
CREATE INDEX IF NOT EXISTS idx_search_history_created ON search_history(created_at DESC);

-- 11. Trigger für automatische Archivierung von Chunks
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

DROP TRIGGER IF EXISTS trigger_cleanup_chunks ON chunks;
CREATE TRIGGER trigger_cleanup_chunks
AFTER UPDATE ON chunks
FOR EACH ROW
EXECUTE FUNCTION cleanup_old_chunks();

-- 12. Übersichtliche Statistikansicht für Adminzwecke
CREATE OR REPLACE VIEW document_stats AS
SELECT 
    d.id,
    d.original_name,
    d.status,
    COUNT(DISTINCT c.id) as chunk_count,
    COUNT(DISTINCT CASE WHEN c.status = 'enhanced' THEN c.id END) as enhanced_chunks,
    COUNT(DISTINCT CASE WHEN c.embedding IS NOT NULL THEN c.id END) as embedded_chunks
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
GROUP BY d.id, d.original_name, d.status;

-- 13. Rechte für n8n oder andere Dienste
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM pg_user WHERE usename = current_user) THEN
        GRANT ALL ON ALL TABLES IN SCHEMA public TO current_user;
        GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO current_user;
    END IF;
END $$;
