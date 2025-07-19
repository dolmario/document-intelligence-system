@echo off
REM semantic-doc-finder Installation Script (Windows)

echo.
echo 🚀 semantic-doc-finder Installation
echo ==========================================

REM Check Requirements
echo 📋 Prüfe Voraussetzungen...

docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker nicht gefunden!
    echo Bitte installiere Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)
echo ✅ Docker gefunden

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose nicht gefunden!
    echo Docker Compose ist normalerweise in Docker Desktop enthalten.
    pause
    exit /b 1
)
echo ✅ Docker Compose gefunden

REM Setup
echo.
echo 🔧 Setup semantic-doc-finder...

REM Run migration
call migration.bat

echo.
echo 📚 Installation abgeschlossen!
echo ==========================================
echo.
echo 🎯 Nächste Schritte:
echo.
echo 1. N8N konfigurieren:
echo    - Öffne: http://localhost:5678
echo    - Login: admin / semantic2024
echo    - Importiere Workflows aus /n8n/workflows/
echo.
echo 2. Upload testen:
echo    - Führe aus: test_upload.bat
echo.
echo 3. Erste Dokumente verarbeiten:
echo    - Kopiere PDFs nach ./data/
echo    - Oder nutze N8N Upload: http://localhost:5678/webhook-test/doc-upload
echo.
echo 4. Search API testen:
echo    - http://localhost:8001/health
echo    - http://localhost:8001/docs
echo.
echo 💡 Hilfe und Dokumentation:
echo    - README.md
echo    - Docker Logs: docker-compose logs -f
echo.
pause
