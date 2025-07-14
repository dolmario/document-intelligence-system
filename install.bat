@echo off
setlocal EnableDelayedExpansion

REM === Document Intelligence System - Windows Installation ===
REM Funktioniert per Doppelklick, CMD und "Als Administrator ausführen"

echo === Document Intelligence System Installation (Windows) ===
echo ============================================================

REM === KRITISCHER FIX: Speichere Original-Verzeichnis ===
set "ORIGINAL_DIR=%CD%"
set "SCRIPT_DIR=%~dp0"

REM === Wechsel ins Skript-Verzeichnis (NICHT System32!) ===
cd /d "%SCRIPT_DIR%"

REM Debug-Info
echo.
echo Debug-Information:
echo - Original-Verzeichnis: %ORIGINAL_DIR%
echo - Skript-Verzeichnis: %SCRIPT_DIR%
echo - Aktuelles Verzeichnis: %CD%

REM === Sicherheitscheck: Sind wir im richtigen Verzeichnis? ===
if not exist "docker-compose.yml" (
    echo.
    echo FEHLER: docker-compose.yml nicht gefunden!
    echo.
    echo Mögliche Ursachen:
    echo 1. Die install.bat liegt nicht im Projekt-Hauptverzeichnis
    echo 2. Das Projekt wurde noch nicht vollständig heruntergeladen
    echo.
    echo Aktuelles Verzeichnis: %CD%
    echo.
    echo Bitte stelle sicher, dass:
    echo - install.bat im gleichen Ordner wie docker-compose.yml liegt
    echo - Alle Projekt-Dateien vorhanden sind
    echo.
    pause
    exit /b 1
)

REM === Admin-Rechte prüfen (OPTIONAL - nicht zwingend nötig) ===
echo.
echo Prüfe Berechtigungen...
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo WARNUNG: Keine Administrator-Rechte!
    echo.
    echo Das Skript läuft trotzdem weiter, aber einige Funktionen
    echo könnten eingeschränkt sein (z.B. Docker-Installation).
    echo.
    echo Für volle Funktionalität:
    echo - Rechtsklick auf install.bat
    echo - "Als Administrator ausführen" wählen
    echo.
    pause
) else (
    echo ✓ Administrator-Rechte verfügbar
)

REM === Erstelle Verzeichnisstruktur ===
echo.
echo Erstelle Verzeichnisse...
if not exist "data" mkdir data
if not exist "indices" mkdir indices
if not exist "indices\json" mkdir indices\json
if not exist "indices\markdown" mkdir indices\markdown
if not exist "logs" mkdir logs
if not exist "n8n\workflows" mkdir n8n\workflows

REM Erstelle .gitkeep Dateien
type nul > data\.gitkeep 2>nul
type nul > indices\.gitkeep 2>nul
type nul > logs\.gitkeep 2>nul

echo ✓ Verzeichnisse erstellt

REM === Python Check ===
echo.
echo Prüfe Python Installation...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Python nicht gefunden!
    echo.
    echo Bitte installiere Python 3.10 oder höher:
    echo 1. Gehe zu https://python.org/downloads
    echo 2. Lade Python herunter und installiere es
    echo 3. WICHTIG: Hake "Add Python to PATH" an!
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%v"
echo ✓ Python gefunden: %PYTHON_VERSION%

REM === Docker Check ===
echo.
echo Prüfe Docker Installation...
docker --version >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Docker nicht gefunden!
    echo.
    echo Bitte installiere Docker Desktop:
    echo 1. Gehe zu https://docker.com/products/docker-desktop
    echo 2. Lade Docker Desktop herunter und installiere es
    echo 3. Starte Docker Desktop und warte bis es bereit ist
    echo.
    pause
    exit /b 1
)

for /f "tokens=3" %%v in ('docker --version 2^>^&1') do set "DOCKER_VERSION=%%v"
echo ✓ Docker gefunden: %DOCKER_VERSION%

REM === Docker läuft? ===
echo.
echo Prüfe ob Docker läuft...
docker ps >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Docker läuft nicht!
    echo.
    echo Bitte:
    echo 1. Starte Docker Desktop
    echo 2. Warte bis das Docker-Symbol grün ist (nicht "Starting...")
    echo 3. Führe dieses Skript erneut aus
    echo.
    pause
    exit /b 1
)
echo ✓ Docker läuft

REM === .env Setup ===
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
            echo PYTORCH_ENABLE_MPS_FALLBACK=1
        ) > ".env"
        echo ✓ .env Datei erstellt
    )
) else (
    echo ✓ .env bereits vorhanden
)

REM === Python Virtual Environment ===
echo.
echo Erstelle Python Virtual Environment...
if not exist "venv" (
    python -m venv venv
    if %errorLevel% neq 0 (
        echo FEHLER: Virtual Environment konnte nicht erstellt werden!
        echo Stelle sicher, dass Python korrekt installiert ist.
        pause
        exit /b 1
    )
    echo ✓ Virtual Environment erstellt
) else (
    echo ✓ Virtual Environment bereits vorhanden
)

REM === Aktiviere venv und installiere Dependencies ===
echo.
echo Installiere Python Dependencies...
echo Das kann einige Minuten dauern...

call venv\Scripts\activate.bat
if %errorLevel% neq 0 (
    echo FEHLER: Virtual Environment konnte nicht aktiviert werden!
    pause
    exit /b 1
)

REM Upgrade pip
python -m pip install --upgrade pip

REM === FIX: PyTorch CPU-Version installieren (vermeidet GPU-Probleme) ===
echo.
echo Installiere PyTorch 2.1.0 (CPU-Version für Kompatibilität)...
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

REM === Requirements installieren ===
if exist "requirements.txt" (
    echo.
    echo Installiere weitere Dependencies...
    REM Installiere ohne Dependencies um Konflikte zu vermeiden
    pip install --no-deps -r requirements.txt
    
    REM Installiere wichtige Pakete einzeln mit Dependencies
    pip install redis fastapi uvicorn spacy pytesseract Pillow pdf2image
    pip install watchdog aiofiles pytest numpy pandas
) else (
    echo ⚠ requirements.txt nicht gefunden - installiere Basis-Pakete
    pip install redis fastapi uvicorn spacy pytesseract Pillow pdf2image watchdog
)

REM === Spacy Modell ===
echo.
echo Lade Spacy Deutsch-Modell...
python -m spacy download de_core_news_sm >nul 2>&1
if %errorLevel% equ 0 (
    echo ✓ Spacy Modell installiert
) else (
    echo ⚠ Spacy Modell konnte nicht geladen werden
)

echo ✓ Python Dependencies installiert

REM === Docker Services starten ===
echo.
echo Starte Docker Services...

REM Stoppe eventuell laufende Services
docker compose down >nul 2>&1

REM === WICHTIG: Verwende CPU-only compose wenn keine GPU ===
set "COMPOSE_FILE=docker-compose.yml"
nvidia-smi >nul 2>&1
if %errorLevel% neq 0 (
    if exist "docker-compose-cpu.yml" (
        set "COMPOSE_FILE=docker-compose-cpu.yml"
        echo Keine GPU gefunden - verwende CPU-Konfiguration
    )
)

REM Starte Services mit der richtigen Datei
docker compose -f %COMPOSE_FILE% up -d
if %errorLevel% neq 0 (
    echo FEHLER: Docker Services konnten nicht gestartet werden!
    echo.
    echo Prüfe die Logs mit: docker compose logs
    pause
    exit /b 1
)

echo ✓ Docker Services gestartet

REM === Warte auf Services ===
echo.
echo Warte 30 Sekunden auf Service-Initialisierung...
timeout /t 30 /nobreak >nul

REM === Service-Status prüfen ===
echo.
echo Prüfe Service-Verfügbarkeit...

docker ps | findstr "redis" >nul && echo ✓ Redis läuft || echo ⚠ Redis Problem
docker ps | findstr "n8n" >nul && echo ✓ N8N läuft || echo ⚠ N8N Problem
docker ps | findstr "search" >nul && echo ✓ Search API läuft || echo ⚠ Search API Problem
docker ps | findstr "webui" >nul && echo ✓ Open WebUI läuft || echo ⚠ WebUI Problem

REM === Test-Dokument erstellen ===
echo.
echo Möchtest du ein Test-Dokument erstellen? (j/n)
set /p create_test=Eingabe: 

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
        echo Dieses Dokument wird automatisch vom Watchdog erkannt.
    ) > "data\test_installation.txt"
    
    echo ✓ Test-Dokument erstellt: data\test_installation.txt
)

REM === Installation abgeschlossen ===
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
echo   1. Lege Dokumente in .\data\ Ordner
echo   2. Öffne http://localhost:8080 für die Suche
echo   3. Konfiguriere N8N Workflows
echo.
echo 🔧 NÜTZLICHE BEFEHLE:
echo   • docker compose logs -f     (Live-Logs anzeigen)
echo   • docker compose ps          (Service-Status)
echo   • docker compose restart     (Services neu starten)
echo.
echo 💡 Bei Problemen:
echo   • Prüfe Logs: docker compose logs
echo   • GitHub Issues: https://github.com/[USERNAME]/document-intelligence-system
echo.

pause
