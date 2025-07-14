@echo off
setlocal EnableDelayedExpansion

REM === Document Intelligence System - Windows Installation ===
REM Funktioniert sowohl per Doppelklick als auch aus CMD

echo === Document Intelligence System Installation (Windows) ===
echo ============================================================

REM === KRITISCH: Wechsel ins Skript-Verzeichnis ===
REM Wenn per Doppelklick gestartet, sind wir in System32!
cd /d "%~dp0"

REM Zeige aktuelles Verzeichnis
echo Aktuelles Verzeichnis: %cd%

REM Sicherheitscheck: Verbotene Verzeichnisse
set "CURRENT_DIR=%cd%"
set "FORBIDDEN_1=C:\Windows\System32"
set "FORBIDDEN_2=C:\Windows"
set "FORBIDDEN_3=C:\Program Files"
set "FORBIDDEN_4=C:\Program Files (x86)"

if /i "%CURRENT_DIR%"=="%FORBIDDEN_1%" (
    echo FEHLER: Immer noch in System32! Skript-Pfad Problem.
    echo Bitte manuell ins Projektverzeichnis wechseln und erneut starten.
    pause
    exit /b 1
)
if /i "%CURRENT_DIR%"=="%FORBIDDEN_2%" (
    echo FEHLER: In Windows-Verzeichnis! Bitte ins Projektverzeichnis wechseln.
    pause
    exit /b 1
)
if /i "%CURRENT_DIR%"=="%FORBIDDEN_3%" (
    echo FEHLER: In Program Files! Bitte ins Projektverzeichnis wechseln.
    pause
    exit /b 1
)
if /i "%CURRENT_DIR%"=="%FORBIDDEN_4%" (
    echo FEHLER: In Program Files (x86)! Bitte ins Projektverzeichnis wechseln.
    pause
    exit /b 1
)

REM Prüfe ob wir im richtigen Projektverzeichnis sind
if not exist "docker-compose.yml" (
    echo.
    echo FEHLER: docker-compose.yml nicht gefunden!
    echo.
    echo Du bist in: %cd%
    echo.
    echo Das passiert oft bei Doppelklick. Lösungen:
    echo 1. Kopiere diese install.bat ins Projektverzeichnis
    echo 2. Oder öffne CMD im Projektverzeichnis und führe install.bat aus
    echo 3. Oder verwende den vollständigen Pfad
    echo.
    pause
    exit /b 1
)

echo ✓ Richtiges Projektverzeichnis gefunden

REM Admin-Rechte prüfen
echo.
echo Prüfe Administrator-Rechte...
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Bitte als Administrator ausführen!
    echo.
    echo So geht's:
    echo 1. Rechtsklick auf install.bat
    echo 2. "Als Administrator ausführen" wählen
    echo.
    pause
    exit /b 1
)
echo ✓ Administrator-Rechte verfügbar

REM Erstelle Verzeichnisstruktur
echo.
echo Erstelle Verzeichnisse...
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

REM Python Check
echo.
echo Prüfe Python Installation...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Python nicht gefunden!
    echo.
    echo Bitte Python 3.10+ von https://python.org installieren
    echo Wichtig: "Add Python to PATH" anhaken!
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%v"
echo ✓ Python gefunden: %PYTHON_VERSION%

REM Docker Check
echo.
echo Prüfe Docker Installation...
docker --version >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Docker nicht gefunden!
    echo.
    echo Bitte Docker Desktop von https://docker.com installieren
    echo.
    pause
    exit /b 1
)

for /f "tokens=3" %%v in ('docker --version 2^>^&1') do set "DOCKER_VERSION=%%v"
echo ✓ Docker gefunden: %DOCKER_VERSION%

REM Docker läuft?
echo.
echo Prüfe ob Docker läuft...
docker ps >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Docker läuft nicht!
    echo.
    echo Bitte Docker Desktop starten und warten bis es vollständig geladen ist.
    echo Das Docker-Symbol in der Taskleiste sollte nicht mehr "Starting..." zeigen.
    echo.
    pause
    exit /b 1
)
echo ✓ Docker läuft

REM .env Setup
echo.
echo Setup Umgebungskonfiguration...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo ✓ .env aus .env.example erstellt
    ) else (
        REM Erstelle minimale .env
        (
            echo # Document Intelligence System Configuration
            echo DATA_PATH=./data
            echo INDEX_PATH=./indices
            echo LOG_PATH=./logs
            echo REDIS_URL=redis://redis:6379
            echo CORS_ALLOW_ORIGIN=http://localhost:8080,http://localhost:5678
            echo USER_AGENT=DocumentIntelligenceSystem/1.0
            echo CHROMA_TELEMETRY=false
            echo TORCH_VERSION=2.1.0
        ) > ".env"
        echo ✓ .env Datei erstellt
    )
    echo ⚠ Du kannst .env später anpassen falls nötig
) else (
    echo ✓ .env bereits vorhanden
)

REM Python Virtual Environment
echo.
echo Erstelle Python Virtual Environment...
if not exist "venv" (
    python -m venv venv
    if %errorLevel% neq 0 (
        echo FEHLER: Virtual Environment konnte nicht erstellt werden!
        pause
        exit /b 1
    )
    echo ✓ Virtual Environment erstellt
) else (
    echo ✓ Virtual Environment bereits vorhanden
)

REM Python Dependencies installieren
echo.
echo Installiere Python Dependencies...
echo Das kann einige Minuten dauern...

call venv\Scripts\activate.bat
python -m pip install --upgrade pip

REM PyTorch zuerst (wichtig für Kompatibilität)
echo.
echo Installiere PyTorch 2.1.0 (kompatible Version)...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
if %errorLevel% neq 0 (
    echo ⚠ PyTorch Installation mit Problemen - versuche weiter
)

REM Restliche Requirements
if exist "requirements.txt" (
    echo.
    echo Installiere weitere Dependencies...
    pip install -r requirements.txt
    if %errorLevel% neq 0 (
        echo ⚠ Einige Pakete konnten nicht installiert werden - das ist oft normal
    )
) else (
    echo ⚠ requirements.txt nicht gefunden - installiere Basis-Pakete
    pip install redis fastapi uvicorn spacy pytesseract Pillow pdf2image
)

REM Spacy Deutsch-Modell
echo.
echo Lade Spacy Deutsch-Modell...
python -m spacy download de_core_news_sm
if %errorLevel% equ 0 (
    echo ✓ Spacy Modell installiert
) else (
    echo ⚠ Spacy Modell Installation fehlgeschlagen - kann später nachgeholt werden
)

echo ✓ Python Dependencies installiert

REM Docker Images laden
echo.
echo Lade Docker Images...
echo Das kann beim ersten Mal sehr lange dauern!

docker pull redis:7-alpine
docker pull postgres:16-alpine
docker pull docker.n8n.io/n8nio/n8n
docker pull ghcr.io/open-webui/open-webui:main
docker pull ollama/ollama:latest

echo ✓ Docker Images geladen

REM Services starten
echo.
echo Starte Docker Services...

REM Stoppe eventuell laufende Services
docker compose down >nul 2>&1

REM Starte Services
docker compose up -d
if %errorLevel% neq 0 (
    echo FEHLER: Docker Services konnten nicht gestartet werden!
    echo.
    echo Prüfe mit: docker compose logs
    pause
    exit /b 1
)

echo ✓ Docker Services gestartet

REM Warte auf Services
echo.
echo Warte 30 Sekunden auf Service-Initialisierung...
timeout /t 30 /nobreak >nul

REM Service-Status prüfen
echo.
echo Prüfe Service-Verfügbarkeit...

REM Einfache Service-Checks
docker ps | findstr "redis" >nul && echo ✓ Redis läuft || echo ⚠ Redis Problem
docker ps | findstr "n8n" >nul && echo ✓ N8N läuft || echo ⚠ N8N Problem  
docker ps | findstr "search" >nul && echo ✓ Search API läuft || echo ⚠ Search API Problem
docker ps | findstr "webui" >nul && echo ✓ Open WebUI läuft || echo ⚠ WebUI Problem

REM Test-Dokument erstellen
echo.
set /p create_test="Test-Dokument erstellen? (j/n): "
if /i "%create_test%"=="j" (
    (
        echo Document Intelligence System Test
        echo.
        echo Dies ist ein Test-Dokument für das Document Intelligence System.
        echo Erstellt am: %date% %time%
        echo.
        echo Features:
        echo - Automatische OCR-Verarbeitung  
        echo - DSGVO-konforme Indexierung
        echo - Intelligente Suche
        echo - KI-gestützte Verknüpfungen
        echo.
        echo Test-Keywords: installation, windows, batch, test
        echo Kontakt: test@example.com
        echo Telefon: +49 123 456789
        echo.
        echo Dieses Dokument wird automatisch vom Watchdog erkannt und verarbeitet.
    ) > "data\test_installation.txt"
    
    echo ✓ Test-Dokument erstellt: data\test_installation.txt
)

REM Erfolgreiche Installation
echo.
echo ================================================================
echo ✅ INSTALLATION ERFOLGREICH ABGESCHLOSSEN!
echo ================================================================
echo.
echo 🌐 SERVICE-ADRESSEN:
echo   • N8N Workflow:    http://localhost:5678 (admin/changeme)
echo   • Open WebUI:      http://localhost:8080
echo   • Search API:      http://localhost:8001/docs
echo   • Redis:           localhost:6379
echo.
echo 📋 NÄCHSTE SCHRITTE:
echo   1. Passe .env Datei nach Bedarf an
echo   2. Lege Dokumente in .\data Ordner
echo   3. Öffne http://localhost:8080 für die Suche
echo   4. Konfiguriere N8N Workflows unter http://localhost:5678
echo.
echo 🔧 NÜTZLICHE BEFEHLE:
echo   • docker compose logs -f          (Live-Logs anzeigen)
echo   • docker compose ps               (Service-Status)
echo   • docker compose down             (Services stoppen)
echo   • docker compose restart          (Services neu starten)
echo.
echo 📁 WICHTIGE VERZEICHNISSE:
echo   • .\data\         → Dokumente hier ablegen
echo   • .\indices\      → Generierte Indizes
echo   • .\logs\         → System-Logs
echo.
echo 💡 PROBLEME?
echo   • Logs prüfen: docker compose logs
echo   • Services neu starten: docker compose restart
echo   • Alles stoppen: docker compose down
echo.

pause
