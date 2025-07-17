#!/bin/bash
# fix_all.sh - Behebt alle gefundenen Fehler

echo "🔧 Fixing all issues..."

# 1. Fix file naming (remove v2 suffixes)
echo "Renaming v2 files..."
if [ -f "agents/ocr/ocr_agent_v2.py" ]; then
    mv agents/ocr/ocr_agent_v2.py agents/ocr/ocr_agent.py
fi
if [ -f "agents/ocr/Dockerfile.v2" ]; then
    mv agents/ocr/Dockerfile.v2 agents/ocr/Dockerfile
fi
if [ -f "agents/ocr/requirements.v2.txt" ]; then
    mv agents/ocr/requirements.v2.txt agents/ocr/requirements.txt
fi

if [ -f "services/search/api_v2.py" ]; then
    mv services/search/api_v2.py services/search/api.py
fi
if [ -f "services/search/Dockerfile.v2" ]; then
    mv services/search/Dockerfile.v2 services/search/Dockerfile
fi
if [ -f "services/search/requirements.v2.txt" ]; then
    mv services/search/requirements.v2.txt services/search/requirements.txt
fi

# 2. Update Dockerfiles to use correct filenames
echo "Updating Dockerfiles..."
if [ -f "agents/ocr/Dockerfile" ]; then
    sed -i 's/ocr_agent_v2\.py/ocr_agent.py/g' agents/ocr/Dockerfile
    sed -i 's/requirements\.v2\.txt/requirements.txt/g' agents/ocr/Dockerfile
fi

if [ -f "services/search/Dockerfile" ]; then
    sed -i 's/api_v2\.py/api.py/g' services/search/Dockerfile
    sed -i 's/requirements\.v2\.txt/requirements.txt/g' services/search/Dockerfile
fi

# 3. Ensure .env has correct database settings
echo "Checking .env..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from template"
fi

# Add DATABASE_URL if missing
if ! grep -q "DATABASE_URL" .env; then
    echo "" >> .env
    echo "# Auto-generated connection URL" >> .env
    echo "DATABASE_URL=postgresql://docintell:docintell123@postgres:5432/document_intelligence" >> .env
    echo "Added DATABASE_URL to .env"
fi

# 4. Clean and rebuild
echo "Rebuilding containers..."
docker compose down
docker compose build --no-cache ocr_agent search_api
docker compose up -d

echo "Waiting for services to start..."
sleep 20

# 5. Run health check
echo "Running health check..."
if [ -f "health_check.py" ]; then
    python health_check.py
else
    echo "health_check.py not found"
fi

echo "✅ All fixes applied!"
echo ""
echo "Next steps:"
echo "1. Activate N8N workflow (see N8N_SETUP.md)"
echo "2. Test with: ./simple_test.sh"
echo "3. Clean old files with: ./cleanup_final.sh"
