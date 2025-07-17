@echo off
echo 🔧 Quick Fix for Document Intelligence System
echo ===========================================

:: Fix 1: Rebuild Search API with correct code
echo Rebuilding Search API...
docker-compose stop search_api
docker-compose build --no-cache search_api
docker-compose up -d search_api

:: Fix 2: Wait for services
echo Waiting for services...
timeout /t 10 >nul

:: Fix 3: Test upload with simple document
echo.
echo Testing system...

:: Prüfen, ob simple_test.sh existiert und ausführbar ist
if exist simple_test.sh (
    echo Running simple_test.sh...
    bash simple_test.sh
) else (
    echo ⚠️ File 'simple_test.sh' not found!
)

echo.
echo ✅ Quick fix applied!
echo.
echo ⚠️  Don't forget to:
echo 1. Activate N8N workflow (see N8N_SETUP.md)
echo 2. Clean up old files (see CLEANUP.md)
echo 3. Change passwords in .env for production

pause