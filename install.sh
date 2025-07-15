#!/bin/bash
# install.sh - Einfache Installation die sofort funktioniert

set -e

echo "📦 Document Intelligence System - Installation"
echo "============================================="

# 1. Basis-Setup
echo "1️⃣ Setup environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env from template"
else
    echo "✅ .env already exists"
fi

# 2. Verzeichnisse erstellen
echo "2️⃣ Creating directories..."
mkdir -p data indices/{json,markdown} logs n8n/workflows
touch data/.gitkeep indices/.gitkeep logs/.gitkeep
echo "✅ Directories created"

# 3. Python venv (optional, für lokale Tests)
if command -v python3 >/dev/null 2>&1; then
    echo "3️⃣ Setting up Python environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        ./venv/bin/pip install --upgrade pip
        ./venv/bin/pip install -r requirements.txt 2>/dev/null || true
        echo "✅ Python environment created"
    else
        echo "✅ Python environment exists"
    fi
fi

# 4. Docker Build & Start
echo "4️⃣ Building Docker containers..."
docker compose build

echo "5️⃣ Starting services..."
# Start in korrekter Reihenfolge
docker compose up -d redis postgres
sleep 10  # Warte auf Datenbanken

docker compose up -d ocr_agent indexer
sleep 5

docker compose up -d search_api n8n ollama open-webui
sleep 10

# 5. Status Check
echo "6️⃣ Checking services..."
docker compose ps

# 6. Test-Dokument erstellen
echo "7️⃣ Creating test document..."
cat > data/welcome.txt << 'EOF'
Welcome to Document Intelligence System!
=======================================

This is a test document to verify the system is working.

Features:
- Automatic OCR processing
- GDPR-compliant indexing  
- Intelligent search
- N8N workflow automation

Test data:
Email: test@example.com
Phone: +49 123 456789

This document should be automatically processed and indexed.
EOF

echo "✅ Test document created: data/welcome.txt"

# 7. Finale Ausgabe
echo "
============================================="
echo "✅ INSTALLATION COMPLETE!"
echo "============================================="
echo ""
echo "🌐 Access Points:"
echo "   • N8N Workflows:  http://localhost:5678 (admin/changeme)"
echo "   • Search UI:      http://localhost:8080"
echo "   • API Docs:       http://localhost:8001/docs"
echo ""
echo "📝 Next Steps:"
echo "   1. Open N8N and import workflow from n8n/workflows/"
echo "   2. Place documents in ./data folder"
echo "   3. Documents will be automatically processed"
echo ""
echo "🛠️ Useful Commands:"
echo "   • View logs:     docker compose logs -f"
echo "   • Stop system:   docker compose down"
echo "   • Restart:       docker compose restart"
echo ""
