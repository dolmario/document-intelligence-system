@echo off
setlocal EnableDelayedExpansion

echo ============================================
echo Document Intelligence System V2 - Installation
echo ============================================

REM Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker not found! Please install Docker Desktop.
    echo Visit: https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running! Please start Docker Desktop.
    pause
    exit /b 1
)

echo [OK] Prerequisites checked

REM Create directories
echo Creating directories...
mkdir data 2>nul
mkdir logs 2>nul
mkdir n8n\workflows 2>nul
type nul > data\.gitkeep 2>nul
type nul > logs\.gitkeep 2>nul

REM Setup environment
echo Setting up environment...
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo [OK] Created .env from template
        echo [!] Please edit .env to set secure passwords!
    ) else (
        echo ERROR: .env.example not found!
        pause
        exit /b 1
    )
) else (
    echo [OK] .env already exists
)

REM Stop existing containers
echo Cleaning up old containers...
docker compose down 2>nul

REM Build containers
echo Building containers (this may take 5-10 minutes)...
docker compose build --no-cache
if %errorlevel% neq 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

REM Start PostgreSQL first
echo Starting PostgreSQL...
docker compose up -d postgres
timeout /t 10 /nobreak >nul

REM Wait for PostgreSQL
echo Waiting for PostgreSQL...
set RETRIES=30
:pgwait
docker compose exec -T postgres pg_isready -U docintell -d document_intelligence >nul 2>&1
if %errorlevel% equ 0 goto pgready
set /a RETRIES-=1
if %RETRIES% leq 0 (
    echo ERROR: PostgreSQL failed to start!
    docker compose logs postgres
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto pgwait

:pgready
echo [OK] PostgreSQL ready

REM Start all services
echo Starting all services...
docker compose up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start services!
    pause
    exit /b 1
)

REM Wait for services
echo Waiting for services to be ready...
timeout /t 20 /nobreak >nul

REM Check services
echo Checking service status...
docker compose ps

REM Load default model
echo Loading default LLM model (mistral)...
docker exec doc-intel-ollama ollama pull mistral 2>nul || echo [!] Model loading failed - you can do this later

echo.
echo ============================================
echo Installation complete!
echo ============================================
echo.
echo Access Points:
echo   - N8N Workflows:  http://localhost:5678 (admin/changeme)
echo   - Search API:     http://localhost:8001
echo   - Open WebUI:     http://localhost:8080
echo   - Qdrant UI:      http://localhost:6333/dashboard
echo.
echo Next Steps:
echo   1. Change default passwords in .env
echo   2. Import workflows from n8n\workflows\
echo   3. Add documents via API or N8N
echo.
echo To add more models:
echo   docker exec doc-intel-ollama ollama pull llama2
echo.
echo To view logs:
echo   docker compose logs -f
echo.
echo To stop:
echo   docker compose down
echo.
pause


