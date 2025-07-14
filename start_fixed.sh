#!/bin/bash
# start_fixed.sh - Garantiert funktionierende Startsequenz

set -e  # Stoppe bei Fehlern

echo "🔧 Starting Document Intelligence System (FIXED VERSION)"

# 1. Stoppe alles und räume auf
echo "Stopping old containers..."
docker compose down -v || true
docker system prune -f

# 2. Erstelle Verzeichnisse
echo "Creating directories..."
mkdir -p data indices/json indices/markdown logs n8n/workflows

# 3. Erstelle .env falls nicht vorhanden
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
POSTGRES_PASSWORD=docintell123
REDIS_URL=redis://redis:6379
USER_AGENT=DocumentIntelligenceSystem/1.0
CORS_ALLOW_ORIGIN=http://localhost:8080,http://localhost:5678
N8N_RUNNERS_ENABLED=true
DB_POSTGRESDB_USER=docintell
DB_POSTGRESDB_PASSWORD=docintell123
DB_POSTGRESDB_DATABASE=document_intelligence
WEBUI_SECRET_KEY=secret-key-change-me
EOF
fi

# 4. Baue ALLE Container neu
echo "Building containers (this may take a while)..."
docker compose build --no-cache

# 5. Starte Services in korrekter Reihenfolge
echo "Starting core services..."
docker compose up -d redis postgres
echo "Waiting for databases to initialize..."
sleep 15

echo "Starting agents..."
docker compose up -d watchdog ocr_agent indexer
sleep 10

echo "Starting API services..."
docker compose up -d search_api n8n ollama
sleep 10

echo "Starting web interface..."
docker compose up -d open-webui

# 6. Zeige Status
echo ""
echo "=== SERVICE STATUS ==="
docker compose ps

# 7. Prüfe Logs
echo ""
echo "=== CHECKING SERVICES ==="

# Redis prüfen
if docker compose exec -T redis redis-cli ping | grep -q PONG; then
    echo "✅ Redis is working"
else
    echo "❌ Redis failed"
fi

# PostgreSQL prüfen
if docker compose exec -T postgres pg_isready -U docintell | grep -q "accepting connections"; then
    echo "✅ PostgreSQL is working"
else
    echo "❌ PostgreSQL failed"
fi

# Agenten prüfen
for service in watchdog ocr_agent indexer search_api n8n; do
    if docker compose ps | grep -q "$service.*Up"; then
        echo "✅ $service is running"
    else
        echo "❌ $service failed"
    fi
done

echo ""
echo "=== ACCESS POINTS ==="
echo "📊 N8N Workflows: http://localhost:5678 (admin/changeme)"
echo "🔍 Search UI: http://localhost:8080"
echo "📚 API Docs: http://localhost:8001/docs"
echo ""
echo "To view logs: docker compose logs -f"
echo "To stop: docker compose down"
