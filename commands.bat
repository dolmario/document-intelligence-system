@echo off
echo Document Intelligence System - Commands
echo ======================================
echo.
echo 1. Install System
echo 2. Start Services  
echo 3. Stop Services
echo 4. View Logs
echo 5. Rebuild Search API
echo 6. Test System
echo 7. Clean Old Files
echo 8. Exit
echo.

choice /c 12345678 /n /m "Select option: "

if errorlevel 8 exit
if errorlevel 7 goto clean
if errorlevel 6 goto test
if errorlevel 5 goto rebuild
if errorlevel 4 goto logs
if errorlevel 3 goto stop
if errorlevel 2 goto start
if errorlevel 1 goto install

:install
call install.bat
pause
goto end

:start
docker compose up -d
echo Services started!
pause
goto end

:stop
docker compose down
echo Services stopped!
pause
goto end

:logs
docker compose logs -f
goto end

:rebuild
docker compose build --no-cache search_api
docker compose up -d search_api
echo Search API rebuilt!
pause
goto end

:test
python health_check.py
pause
goto end

:clean
echo Cleaning old files...
del /q fix_*.py fix_*.bat start_*.py start_*.bat check_*.py 2>nul
rmdir /s /q agents\watchdog 2>nul
rmdir /s /q indices 2>nul
echo Cleanup done!
pause
goto end

:end
