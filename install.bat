from pathlib import Path

# Inhalt der neuen install.bat mit robuster Docker-Logik und GPU-Prüfung
install_bat_content = r"""@echo off
setlocal EnableDelayedExpansion

REM === Document Intelligence System - Windows Installation ===
echo === Document Intelligence System Installation (Windows) ===
echo ============================================================

set "ORIGINAL_DIR=%CD%"
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo Debug-Information:
echo - Original-Verzeichnis: %ORIGINAL_DIR%
echo - Skript-Verzeichnis: %SCRIPT_DIR%
echo - Aktuelles Verzeichnis: %CD%

if not exist "docker-compose.yml" (
    echo.
    echo FEHLER: docker-compose.yml nicht gefunden!
    echo Bitte stelle sicher, dass diese Datei im Projektverzeichnis liegt.
    pause
    exit /b 1
)

echo.
echo Prüfe Berechtigungen...
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNUNG: Keine Admin-Rechte. Fortfahren...
) else (
    echo ✓ Administratorrechte erkannt.
)

echo.
echo Erstelle Verzeichnisse...
mkdir data 2>nul
mkdir indices\json 2>nul
mkdir indices\markdown 2>nul
mkdir logs 2>nul
mkdir n8n\workflows 2>nul
type nul > data\.gitkeep 2>nul
type nul > indices\.gitkeep 2>nul
type nul > logs\.gitkeep 2>nul
echo ✓ Verzeichnisse erstellt

echo.
echo Prüfe Python Installation...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Python nicht gefunden!
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%v"
echo ✓ Python gefunden: %PYTHON_VERSION%

echo.
echo Prüfe Docker Installation...
docker --version >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Docker nicht gefunden!
    pause
    exit /b 1
)
echo ✓ Docker erkannt

echo.
echo Prüfe ob Docker läuft...
docker ps >nul 2>&1
if %errorLevel% neq 0 (
    echo FEHLER: Docker läuft nicht!
    pause
    exit /b 1
)
echo ✓ Docker läuft

echo.
echo Setup .env Datei...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo ✓ .env erstellt aus .env.example
    ) else (
        echo CHROMA_TELEMETRY=false> .env
        echo ✓ Minimale .env erstellt
    )
) else (
    echo ✓ .env bereits vorhanden
)

echo.
echo Python Virtual Environment...
if not exist "venv" (
    python -m venv venv
    if %errorLevel% neq 0 (
        echo FEHLER: venv konnte nicht erstellt werden.
        pause
        exit /b 1
    )
    echo ✓ venv erstellt
)
call venv\Scripts\activate.bat
if %errorLevel% neq 0 (
    echo FEHLER: Konnte venv nicht aktivieren
    pause
    exit /b 1
)

echo.
echo Installiere pip + dependencies...
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt 2>nul

echo.
echo Installiere Standardpakete (OCR, NLP etc.)...
pip install redis fastapi uvicorn spacy pytesseract Pillow pdf2image watchdog aiofiles pytest numpy pandas >nul
python -m spacy download de_core_news_sm >nul

echo ✓ Python-Pakete installiert

REM === GPU-Erkennung ===
set "COMPOSE_FILE=docker-compose.yml"
set "PATH=%PATH%;C:\Program Files\NVIDIA Corporation\NVSMI"
nvidia-smi >nul 2>&1
if %errorLevel% neq 0 (
    if exist "docker-compose-cpu.yml" (
        set "COMPOSE_FILE=docker-compose-cpu.yml"
        echo [INFO] Keine GPU erkannt – verwende CPU-Setup
    )
) else (
    echo [INFO] NVIDIA GPU erkannt – verwende Standard-Setup
)

echo.
echo [9/10] Docker Build vorbereiten...
docker compose -f %COMPOSE_FILE% build --progress plain
if %errorLevel% neq 0 (
    echo FEHLER: Build fehlgeschlagen!
    pause
    exit /b 1
)

echo.
echo [10/10] Starte Docker-Services...
docker compose -f %COMPOSE_FILE% up -d
if %errorLevel% neq 0 (
    echo FEHLER: Container konnten nicht gestartet werden!
    pause
    exit /b 1
)

echo.
echo ✓ Installation abgeschlossen!
echo Zugriff auf:
echo - N8N:       http://localhost:5678
echo - OpenWebUI: http://localhost:8080
echo - SearchAPI: http://localhost:8001
echo.
pause
"""

# Speicherort für install.bat
output_path = Path("/mnt/data/install.bat")
output_path.write_text(install_bat_content, encoding="utf-8")
output_path

