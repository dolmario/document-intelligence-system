# N8N Webhook Setup

## ⚠️ Wichtig: Webhook muss aktiviert werden!

### 1. Workflow importieren
1. Öffne N8N: http://localhost:5678
2. Login: admin / changeme (oder deine Credentials aus .env)
3. Klicke auf "Workflows" → "Import from File"
4. Wähle: `n8n/workflows/document_processing_v2.json`

### 2. PostgreSQL Credentials einrichten
1. Gehe zu "Credentials" → "New"
2. Wähle "Postgres"
3. Fülle aus:
   - **Host**: `postgres`
   - **Database**: `document_intelligence`
   - **User**: `docintell`
   - **Password**: `docintell123` (oder aus .env)
   - **Port**: `5432`
4. Speichern als "DocIntel DB"

### 3. Workflow aktivieren
1. Öffne den importierten Workflow
2. **WICHTIG**: Klicke oben rechts auf den Toggle-Switch "Inactive" → "Active"
3. Der Webhook ist jetzt unter `/webhook/doc-upload` erreichbar

### 4. Test Upload via Webhook
```bash
curl -X POST http://localhost:5678/webhook/doc-upload \
  -H "Content-Type: application/json" \
  -d '{
    "fileName": "test.pdf",
    "fileType": "pdf",
    "fileContent": "SGVsbG8gV29ybGQ=",
    "metadata": {"source": "test"}
  }'
```

### 5. Production vs Test URLs
- **Test URL** (immer aktiv): http://localhost:5678/webhook-test/doc-upload
- **Production URL** (nur wenn Workflow aktiv): http://localhost:5678/webhook/doc-upload

## 🔧 Troubleshooting

### "Webhook not registered" Fehler
→ Workflow ist nicht aktiviert! Toggle-Switch auf "Active" setzen.

### "Connection refused" 
→ PostgreSQL Credentials prüfen

### Workflow startet nicht
→ N8N neu starten: `docker compose restart n8n`
