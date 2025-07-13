@echo off
title Document Intelligence System Setup
color 0A
echo ================================================
echo    Document Intelligence System Installer
echo ================================================

REM Pfade
set REPO=https://github.com/dolmario/document-intelligence-system.git
set FOLDER=document-intelligence-system

REM 1. Git prüfen
where git >nul 2>&1
IF ERRORLEVEL 1 (
    echo [FEHLER] Git ist nicht installiert. Bitte installiere es von https://git-scm.com
    pause
    exit /b
) ELSE (
    echo [OK] Git gefunden
)

REM 2. Docker prüfen
where docker >nul 2>&1
IF ERRORLEVEL 1 (
    echo [FEHLER] Docker ist nicht installiert. Bitte installiere Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b
) ELSE (
    echo [OK] Docker gefunden
)

REM 3. Repo klonen
IF EXIST "%FOLDER%" (
    echo [WARNUNG] Ordner "%FOLDER%" existiert bereits – Klonen übersprungen
) ELSE (
    git clone %REPO%
    IF ERRORLEVEL 1 (
        echo [FEHLER] Konnte Repository nicht klonen
        pause
        exit /b
    )
    echo [OK] Repository geklont
)

cd %FOLDER%

REM 4. .env einrichten
IF EXIST ".env" (
    echo [WARNUNG] .env existiert bereits
) ELSE (
    copy .env.example .env
    echo [OK] .env Datei erstellt
)

REM 5. Docker Compose starten
echo [INFO] Starte Docker Services...
docker compose up -d

IF ERRORLEVEL 1 (
    echo [FEHLER] Docker Compose konnte nicht gestartet werden
    pause
    exit /b
)

REM 6. Hinweis auf Dienste
echo.
echo Dienste gestartet:
echo - Search API:   http://localhost:8001/docs
echo - N8N:          http://localhost:5678
echo - WebUI:        http://localhost:8080
echo.
echo Logs: docker compose logs -f
echo Stoppen: docker compose down

pause
