@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

echo === Document Intelligence System Installation (Windows) ===
echo ============================================================

REM Setze Startverzeichnis
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo Prüfe docker-compose.yml...
if not exist "docker-compose.yml" (
    echo FEHLER: docker-compose.yml nicht gefunden!
    echo Bitte führe dieses Skript im Projekt-Hauptverzeichnis aus.
    pause
    exit /b 1
)

REM === Python prüfen
echo.
echo Prüfe Python...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo Python nicht gefunden! Bitte installiere Python 3.10+
    pause
    exit /b 1
)

REM === Docker prüfen
echo.
echo Prüfe Docker...
docker --version >nul 2>&1
if %errorLevel% neq 0 (
    echo Docker nicht gefunden! Bitte installiere Docker Desktop
    pause
    exit /b 1
)

REM === Verzeichnisse erstellen
echo.
echo Erstelle Ordnerstruktur...
mkdir data >nul 2>&1
mkdir indices\json >nul 2>&1
mkdir indices\markdown >nul 2>&1
mkdir logs >nul 2>&1
mkdir n8n\workflows >nul 2>&1
type nul > data\.gitkeep
type nul > indices\.gitkeep
type nul > logs\.gitkeep

echo.
echo Erstelle .env...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo .env aus Vorlage erstellt.
    ) else (
        echo DATA_PATH=./data > .env
        echo INDEX_PATH=./indices >> .env
        echo LOG_PATH=./logs >> .env
        echo REDIS_URL=redis://redis:6379 >> .env
        echo .env Datei minimal erstellt.
    )
) else (
    echo .env Datei bereits vorhanden.
)

REM === Virtualenv
echo.
echo Erstelle virtuelles Python Environment...
if not exist "venv" (
    python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

REM === Install requirements
echo.
if exist "requirements.txt" (
    echo Installiere Python-Abhängigkeiten...
    pip install -r requirements.txt
) else (
    echo requirements.txt fehlt – überspringe...
)

echo.
echo Lade Spacy Modell...
python -m spacy download de_core_news_sm

REM === Docker Compose
echo.
echo Starte Docker Compose...
docker compose down >nul 2>&1
docker compose up -d

echo.
echo Warte auf Services...
timeout /t 15 /nobreak >nul

echo.
echo Prüfung abgeschlossen – Dienste gestartet!
echo Öffne http://localhost:8080 im Browser.
echo.
pause
