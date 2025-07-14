@echo off
echo Setting up Ollama models...

:: Wait for Ollama to be ready
echo Waiting for Ollama...
:wait_ollama
timeout /t 2 /nobreak >nul
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorLevel% neq 0 goto wait_ollama

echo Ollama ready. Loading models...

:: Load recommended models
echo Loading mistral...
docker exec document-intelligence-system-ollama-1 ollama pull mistral

echo Loading llama3...
docker exec document-intelligence-system-ollama-1 ollama pull llama3

echo Loading phi...
docker exec document-intelligence-system-ollama-1 ollama pull phi

echo Models loaded!
pause
