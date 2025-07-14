@echo off
echo === Document Intelligence System Installation (Windows) ===
echo =========================================================

:: Check for Admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Bitte als Administrator ausfuehren!
    pause
    exit /b 1
)

:: Create directories first
echo.
echo Creating directory structure...
if not exist "data" mkdir data
if not exist "indices" mkdir indices
if not exist "indices\json" mkdir indices\json
if not exist "indices\markdown" mkdir indices\markdown
if not exist "logs" mkdir logs
if not exist "n8n\workflows" mkdir n8n\workflows

:: Create .gitkeep files
echo. > data\.gitkeep
echo. > indices\.gitkeep
echo. > logs\.gitkeep

echo ✓ Verzeichnisse erstellt

:: Check Python
echo.
echo Checking Python...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Python nicht gefunden. Bitte installieren.
    pause
    exit /b 1
)
echo ✓ Python gefunden

:: Check Docker
echo.
echo Checking Docker...
docker --version >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Docker nicht gefunden. Bitte Docker Desktop installieren.
    pause
    exit /b 1
)
echo ✓ Docker gefunden

:: Check if Docker is running
docker ps >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Docker laeuft nicht. Bitte Docker Desktop starten.
    pause
    exit /b 1
)
echo ✓ Docker laeuft

:: Setup environment
echo.
echo Setting up environment...
if not exist ".env" (
    copy .env.example .env
    echo ✓ .env Datei erstellt
    echo ⚠ Bitte .env Datei anpassen!
) else (
    echo ⚠ .env existiert bereits
)

:: Python Virtual Environment
echo.
echo Creating Python virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✓ Virtual environment erstellt
)

:: Install Python dependencies
echo.
echo Installing Python dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download de_core_news_sm
echo ✓ Python Dependencies installiert

:: Pull Docker images individually to avoid network issues
echo.
echo Pulling Docker images (this may take a while)...
echo.

echo Pulling Redis...
docker pull redis:7-alpine

echo Pulling N8N...
docker pull docker.n8n.io/n8nio/n8n

echo Pulling Postgres...
docker pull postgres:16-alpine

echo Pulling Ollama (large image - may take longer)...
docker pull ollama/ollama:latest

echo Pulling Open WebUI...
docker pull ghcr.io/open-webui/open-webui:main

echo ✓ Docker images pulled

:: Start services
echo.
echo Starting services...
docker compose up -d

:: Wait for services
echo.
echo Waiting for services to start (30 seconds)...
timeout /t 30 /nobreak >nul

:: Check services
echo.
echo Checking services...
curl -s http://localhost:6379 >nul 2>&1
if %errorLevel% eq 0 (
    echo ✓ Redis running
) else (
    echo ⚠ Redis not reachable
)

curl -s http://localhost:5678 >nul 2>&1
if %errorLevel% eq 0 (
    echo ✓ N8N running at http://localhost:5678
) else (
    echo ⚠ N8N not reachable
)

curl -s http://localhost:8001 >nul 2>&1
if %errorLevel% eq 0 (
    echo ✓ Search API running at http://localhost:8001
) else (
    echo ⚠ Search API not reachable
)

curl -s http://localhost:8080 >nul 2>&1
if %errorLevel% eq 0 (
    echo ✓ Open WebUI running at http://localhost:8080
) else (
    echo ⚠ Open WebUI not reachable
)

:: Create test document
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

:: Final message
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
echo   1. Edit .env file
echo   2. Add documents to .\data
echo   3. Open WebUI for search
echo.
echo Commands:
echo   docker compose logs -f    (view logs)
echo   docker compose down       (stop services)
echo   docker compose ps         (check status)
echo.
pause
