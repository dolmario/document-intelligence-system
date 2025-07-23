# Chunks Tabellen Schema

## **chunks**

| Feld | Typ | Default |
|------|-----|---------|
| `id` | UUID | uuid_generate_v4() |
| `document_id` | UUID | - |
| `chunk_index` | INTEGER | - |
| `content` | TEXT | - |
| `content_type` | VARCHAR(50) | 'text' |
| `page_number` | INTEGER | NULL |
| `position` | JSONB | NULL |
| `status` | VARCHAR(50) | 'raw' |
| `embedding` | FLOAT[] | NULL |
| `metadata` | JSONB | `{}` |
| `created_at` | TIMESTAMP | CURRENT_TIMESTAMP |
| `enhanced_at` | TIMESTAMP | NULL |

## **enhanced_chunks**

| Feld | Typ | Default |
|------|-----|---------|
| `id` | UUID | uuid_generate_v4() |
| `document_id` | UUID | - |
| `original_chunk_id` | UUID | - |
| `chunk_index` | INTEGER | - |
| `enhanced_content` | TEXT | - |
| `original_content` | TEXT | NULL |
| `categories` | JSONB | `[]` |
| `extracted_metadata` | JSONB | `{}` |
| `detected_references` | JSONB | `[]` |
| `key_topics` | JSONB | `[]` |
| `content_type` | VARCHAR(50) | 'text' |
| `page_number` | INTEGER | NULL |
| `source_file_path` | TEXT | NULL |
| `source_drive_link` | TEXT | NULL |
| `source_repository` | TEXT | NULL |
| `enhancement_model` | VARCHAR(50) | NULL |
| `confidence_score` | FLOAT | 0.5 |
| `processing_time` | INTEGER | NULL |
| `quality_score` | FLOAT | 0.5 |
| `manual_review_needed` | BOOLEAN | FALSE |
| `created_at` | TIMESTAMP | NOW() |
| `updated_at` | TIMESTAMP | NOW() |
