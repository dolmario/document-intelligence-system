@echo off
setlocal EnableDelayedExpansion
set LOGFILE=testlog.txt
echo Testing Document Intelligence System > %LOGFILE%

REM Test service
call :test_service "Search API" http://localhost:8001/ 200
call :test_service "N8N" http://localhost:5678/ 200
call :test_service "Open WebUI" http://localhost:8080/ 200
call :test_service "Qdrant" http://localhost:6333/ 200
call :test_service "Ollama" http://localhost:11434/ 200

REM Upload test
echo Testing document upload... >> %LOGFILE%
echo.
echo 📄 Uploading document...
curl -s -X POST http://localhost:8001/upload ^
  -H "Content-Type: application/json" ^
  -d "{\"name\":\"test.txt\",\"content\":\"This is a test document.\"}" > upload_response.json

type upload_response.json >> %LOGFILE%
findstr /C:"document_id" upload_response.json >nul
if %errorlevel%==0 (
    echo ✅ Upload successful
    echo Upload successful >> %LOGFILE%
) else (
    echo ❌ Upload failed
    echo Upload failed >> %LOGFILE%
)

REM Search test
timeout /t 5 >nul
echo.
echo 🔍 Performing search...
curl -s -X POST http://localhost:8001/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"test document\",\"limit\":10}" > search_response.json

type search_response.json >> %LOGFILE%
findstr /C:"[" search_response.json >nul
if %errorlevel%==0 (
    echo ✅ Search working
    echo Search working >> %LOGFILE%
) else (
    echo ❌ Search failed
    echo Search failed >> %LOGFILE%
)

REM Show container status
echo. >> %LOGFILE%
echo 📊 Docker container status:
docker compose ps >> %LOGFILE%

echo.
echo Done. See %LOGFILE% for full output.
pause
exit /b

:test_service
set NAME=%~1
set URL=%~2
set EXPECTED=%~3

echo Testing %NAME%...
echo Testing %NAME%... >> %LOGFILE%
for /f %%i in ('curl -s -o nul -w "%%{http_code}" %URL%') do set RESPONSE=%%i

if "!RESPONSE!"=="%EXPECTED%" (
    echo ✅ %NAME% OK
    echo %NAME% OK >> %LOGFILE%
) else (
    echo ❌ %NAME% FAILED (HTTP !RESPONSE!)
    echo %NAME% FAILED (HTTP !RESPONSE!) >> %LOGFILE%
)
goto :eof
