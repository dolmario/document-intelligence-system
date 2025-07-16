@echo off
setlocal enabledelayedexpansion

echo 📦 Document Intelligence System V2 - Installation
echo ==============================================

:: 1. Create directories
echo Creating directories...
mkdir data
mkdir logs
mkdir n8n\workflows
type nul > data\.gitkeep
type nul > logs\.gitkeep

:: 2. Copy env file
if not exist .env (
    copy .env.example .env
    echo ✅ Created .env - Please edit with your settings
) else (
    echo ✅ .env exists
)

:: 3. Build and start
echo Building containers...
docker compose build

echo Starting services...
docker compose up -d

:: 4. Wait for services
echo Waiting for services to be ready...
timeout /t 20 > nul

:: 5. Initialize Ollama model
echo Loading default model (mistral)...
docker exec doc-intel-ollama ollama pull mistral || echo Model loading can be done later

:: 6. Status
echo:
echo ✅ Installation complete!
echo:
echo 🌐 Access Points:
echo    • N8N Workflows:  http://localhost:5678 (admin/changeme)
echo    • Search API:     http://localhost:8001
echo    • Open WebUI:     http://localhost:8080
echo    • Qdrant:         http://localhost:6333/dashboard
echo:
echo 📝 Next Steps:
echo    1. Change passwords in .env
echo    2. Import n8n workflows from n8n\workflows\
echo    3. Add documents via API or n8n
echo:
echo 🤖 To add more models:
echo    docker exec doc-intel-ollama ollama pull <model-name>
echo:

pause


