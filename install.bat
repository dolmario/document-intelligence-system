@echo off
setlocal EnableDelayedExpansion EnableExtensions

REM === Document Intelligence System - Windows Batch Installation ===
REM Version 2.0 - Umfassende Fixes und Validierungen

REM Initialisierung
set "SCRIPT_NAME=install.bat"
set "LOG_FILE=install_batch.log"
set "ERROR_LEVEL=0"
set "INSTALL_START_TIME=%date% %time%"

REM Lösche altes Log
if exist "%LOG_FILE%" del "%LOG_FILE%"

REM === LOGGING FUNKTIONEN ===
:LOG
echo [%date% %time%] %~1 | tee -a "%LOG_FILE%"
echo [%date% %time%] %~1
goto :eof

:LOG_ERROR
echo [%date% %time%] [ERROR] %~1 | tee -a "%LOG_FILE%"
echo [%date% %time%] [ERROR] %~1
set "ERROR_LEVEL=1"
goto :eof

:LOG_SUCCESS
echo [%date% %time%] [SUCCESS] %~1 | tee -a "%LOG_FILE%"
echo [%date% %time%] [SUCCESS] %~1
goto :eof

:LOG_WARN
echo [%date% %time%] [WARN] %~1 | tee -a "%LOG_FILE%"
echo [%date% %time%] [WARN] %~1
goto :eof

REM === HEADER FUNKTION ===
:SHOW_HEADER
echo.
echo ================================================================
echo %~1
echo ================================================================
call :LOG "=== %~1 ==="
goto :eof

REM === FEHLER HANDLER ===
:EXIT_WITH_ERROR
call :LOG_ERROR "%~1"
echo.
echo INSTALLATION FEHLGESCHLAGEN!
echo Siehe %LOG_FILE% für Details.
echo.
pause
exit /b 1

REM === SICHERHEITSCHECKS ===
:CHECK_SECURITY
call :SHOW_HEADER "SICHERHEITSCHECKS"

REM Admin-Rechte prüfen
call :LOG "Prüfe Administrator-Rechte..."
net session >nul 2>&1
if %errorLevel% neq 0 (
    call :EXIT_WITH_ERROR "FEHLER: Bitte als Administrator ausführen!"
)
call :LOG_SUCCESS "Administrator-Rechte verfügbar"

REM Verzeichnis-Validierung
set "CURRENT_DIR=%cd%"
call :LOG "Aktuelles Verzeichnis: %CURRENT_DIR%"

REM Verbotene Verzeichnisse
set "FORBIDDEN_1=C:\Windows\System32"
set "FORBIDDEN_2=C:\Windows"
set "FORBIDDEN_3=C:\Program Files"
set "FORBIDDEN_4=C:\Program Files (x86)"

if /i "%CURRENT_DIR%"=="%FORBIDDEN_1%" call :EXIT_WITH_ERROR "Installation aus System32 nicht erlaubt!"
if /i "%CURRENT_DIR%"=="%FORBIDDEN_2%" call :EXIT_WITH_ERROR "Installation aus Windows-Verzeichnis nicht erlaubt!"
if /i "%CURRENT_DIR%"=="%FORBIDDEN_3%" call :EXIT_WITH_ERROR "Installation aus Program Files nicht erlaubt!"
if /i "%CURRENT_DIR%"=="%FORBIDDEN_4%" call :EXIT_WITH_ERROR "Installation aus Program Files (x86) nicht erlaubt!"

REM Projekt-Verzeichnis validieren
if not exist "docker-compose.yml" (
    call :EXIT_WITH_ERROR "docker-compose.yml nicht gefunden. Bitte aus Projekt-Root ausführen!"
)

call :LOG_SUCCESS "Sicherheitschecks bestanden"
goto :eof

REM === SOFTWARE PRÜFUNG ===
:CHECK_SOFTWARE
call :SHOW_HEADER "SOFTWARE-KOMPONENTEN PRÜFEN"

REM Python Check
call :LOG "Prüfe Python Installation..."
python --version >nul 2>&1
if %errorLevel% neq 0 (
    call :EXIT_WITH_ERROR "Python nicht gefunden. Bitte von python.org installieren!"
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%v"
call :LOG_SUCCESS "Python gefunden: %PYTHON_VERSION%"

REM Python Version Check (vereinfacht)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set "PY_
