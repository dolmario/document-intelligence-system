# Chunks Tabellen Schema

## **chunks**

| Feld | Typ | Default | N8N Zugriff |
|------|-----|---------|-------------|
| `id` | UUID | uuid_generate_v4() | `$json.id` |
| `document_id` | UUID | - | `$json.document_id` |
| `chunk_index` | INTEGER | - | `$json.chunk_index` |
| `content` | TEXT | - | `$json.content` |
| `content_type` | VARCHAR(50) | 'text' | `$json.content_type` |
| `page_number` | INTEGER | NULL | `$json.page_number` |
| `position` | JSONB | NULL | `$json.position` |
| `status` | VARCHAR(50) | 'raw' | `$json.status` |
| `embedding` | FLOAT[] | NULL | `$json.embedding` |
| `metadata` | JSONB | `{}` | `$json.metadata` |
| `created_at` | TIMESTAMP | CURRENT_TIMESTAMP | `$json.created_at` |
| `enhanced_at` | TIMESTAMP | NULL | `$json.enhanced_at` |

## **enhanced_chunks**

| Feld | Typ | Default | N8N Zugriff |
|------|-----|---------|-------------|
| `id` | UUID | uuid_generate_v4() | `$json.id` |
| `document_id` | UUID | - | `$json.document_id` |
| `original_chunk_id` | UUID | - | `$json.original_chunk_id` |
| `chunk_index` | INTEGER | - | `$json.chunk_index` |
| `enhanced_content` | TEXT | - | `$json.enhanced_content` |
| `original_content` | TEXT | NULL | `$json.original_content` |
| `categories` | JSONB | `[]` | `$json.categories` |
| `extracted_metadata` | JSONB | `{}` | `$json.extracted_metadata` |
| `detected_references` | JSONB | `[]` | `$json.detected_references` |
| `key_topics` | JSONB | `[]` | `$json.key_topics` |
| `content_type` | VARCHAR(50) | 'text' | `$json.content_type` |
| `page_number` | INTEGER | NULL | `$json.page_number` |
| `source_file_path` | TEXT | NULL | `$json.source_file_path` |
| `source_drive_link` | TEXT | NULL | `$json.source_drive_link` |
| `source_repository` | TEXT | NULL | `$json.source_repository` |
| `enhancement_model` | VARCHAR(50) | NULL | `$json.enhancement_model` |
| `confidence_score` | FLOAT | 0.5 | `$json.confidence_score` |
| `processing_time` | INTEGER | NULL | `$json.processing_time` |
| `quality_score` | FLOAT | 0.5 | `$json.quality_score` |
| `manual_review_needed` | BOOLEAN | FALSE | `$json.manual_review_needed` |
| `created_at` | TIMESTAMP | NOW() | `$json.created_at` |
| `updated_at` | TIMESTAMP | NOW() | `$json.updated_at` |
