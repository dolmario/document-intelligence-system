@echo off
REM === Einfache funktionierende Installation ===

echo === Document Intelligence System Installation (Windows) ===
echo ============================================================

REM Check for Admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Bitte als Administrator ausfuehren!
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "docker-compose.yml" (
    echo FEHLER: docker-compose.yml nicht gefunden. Bitte aus Projekt-Root ausfuehren.
    pause
    exit /b 1
)

echo.
echo Creating directory structure...
if not exist "data" mkdir data
if not exist "indices" mkdir indices
if not exist "indices\json" mkdir indices\json
if not exist "indices\markdown" mkdir indices\markdown
if not exist "logs" mkdir logs
if not exist "n8n\workflows" mkdir n8n\workflows

echo. > data\.gitkeep
echo. > indices\.gitkeep
echo. > logs\.gitkeep

echo ✓ Verzeichnisse erstellt

REM Check Python
echo.
echo Checking Python...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Python nicht gefunden. Bitte installieren.
    pause
    exit /b 1
)
echo ✓ Python gefunden

REM Check Docker
echo.
echo Checking Docker...
docker --version >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Docker nicht gefunden. Bitte Docker Desktop installieren.
    pause
    exit /b 1
)
echo ✓ Docker gefunden

REM Check if Docker is running
docker ps >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Docker laeuft nicht. Bitte Docker Desktop starten.
    pause
    exit /b 1
)
echo ✓ Docker laeuft

REM Setup environment
echo.
echo Setting up environment...
if not exist ".env" (
    copy .env.example .env
    echo ✓ .env Datei erstellt
) else (
    echo .env existiert bereits
)

REM Python Virtual Environment
echo.
echo Creating Python virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✓ Virtual environment erstellt
)

REM Install Python dependencies
echo.
echo Installing Python dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install -r requirements.txt
python -m spacy download de_core_news_sm
echo ✓ Python Dependencies installiert

REM Pull Docker images
echo.
echo Pulling Docker images...
docker pull redis:7-alpine
docker pull docker.n8n.io/n8nio/n8n
docker pull postgres:16-alpine
docker pull ollama/ollama:latest
docker pull ghcr.io/open-webui/open-webui:main
echo ✓ Docker images pulled

REM Start services
echo.
echo Starting services...
docker compose up -d

REM Wait for services
echo.
echo Waiting for services to start (30 seconds)...
timeout /t 30 /nobreak >nul

REM Check services
echo.
echo Checking services...
echo Services should be available at:
echo   - N8N:         http://localhost:5678 (admin/changeme)
echo   - Open WebUI:  http://localhost:8080
echo   - Search API:  http://localhost:8001/docs
echo   - Redis:       localhost:6379

REM Create test document
echo.
set /p create_test="Create test document? (y/n): "
if /i "%create_test%"=="y" (
    echo Dies ist ein Test-Dokument fuer das Document Intelligence System. > data\test_document.txt
    echo. >> data\test_document.txt
    echo Es enthaelt verschiedene Abschnitte und Informationen. >> data\test_document.txt
    echo. >> data\test_document.txt
    echo Features: >> data\test_document.txt
    echo - Automatische Texterkennung >> data\test_document.txt
    echo - DSGVO-konforme Verarbeitung >> data\test_document.txt
    echo - Intelligente Verknuepfungen >> data\test_document.txt
    echo. >> data\test_document.txt
    echo Kontakt: test@example.com >> data\test_document.txt
    echo Tel: +49 123 456789 >> data\test_document.txt
    
    echo ✓ Test document created
)

REM Final message
echo.
echo =========================================
echo Installation completed!
echo =========================================
echo.
echo Services:
echo   - N8N:         http://localhost:5678 (admin/changeme)
echo   - Open WebUI:  http://localhost:8080
echo   - Search API:  http://localhost:8001/docs
echo   - Redis:       localhost:6379
echo.
echo Next steps:
echo   1. Edit .env file if needed
echo   2. Add documents to .\data
echo   3. Open WebUI for search
echo.
echo Commands:
echo   docker compose logs -f    (view logs)
echo   docker compose down       (stop services)
echo   docker compose ps         (check status)
echo.
pause
