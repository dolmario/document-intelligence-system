# Einfache funktionierende PowerShell Installation

Write-Host "`n=== Document Intelligence System Installation (PowerShell) ===" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan

# Admin Check
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole("Administrator")) {
    Write-Host "FEHLER: Bitte als Administrator ausfuehren!" -ForegroundColor Red
    Read-Host "Druecke Enter zum Beenden"
    exit 1
}

# Check if in right directory
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "FEHLER: docker-compose.yml nicht gefunden. Bitte aus Projekt-Root ausfuehren." -ForegroundColor Red
    Read-Host "Druecke Enter zum Beenden"
    exit 1
}

Write-Host "`nErstelle Verzeichnisse..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data", "indices/json", "indices/markdown", "logs", "n8n/workflows" | Out-Null
" " | Set-Content "data/.gitkeep"
" " | Set-Content "indices/.gitkeep"
" " | Set-Content "logs/.gitkeep"
Write-Host "✓ Verzeichnisse erstellt" -ForegroundColor Green

# Python Check
Write-Host "`nPruefe Python..." -ForegroundColor Yellow
try {
    $null = & python --version
    Write-Host "✓ Python gefunden" -ForegroundColor Green
} catch {
    Write-Host "FEHLER: Python nicht installiert oder nicht im PATH" -ForegroundColor Red
    Read-Host "Druecke Enter zum Beenden"
    exit 1
}

# Docker Check
Write-Host "`nPruefe Docker..." -ForegroundColor Yellow
try {
    $null = & docker --version
    Write-Host "✓ Docker gefunden" -ForegroundColor Green
} catch {
    Write-Host "FEHLER: Docker nicht installiert" -ForegroundColor Red
    Read-Host "Druecke Enter zum Beenden"
    exit 1
}

# Docker running?
try {
    $null = & docker ps
    Write-Host "✓ Docker laeuft" -ForegroundColor Green
} catch {
    Write-Host "FEHLER: Docker laeuft nicht. Bitte Docker Desktop starten." -ForegroundColor Red
    Read-Host "Druecke Enter zum Beenden"
    exit 1
}

# .env setup
Write-Host "`nSetup .env..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✓ .env Datei erstellt" -ForegroundColor Green
} else {
    Write-Host ".env existiert bereits" -ForegroundColor Yellow
}

# Python venv
Write-Host "`nErstelle Python Virtual Environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    & python -m venv venv
    Write-Host "✓ Virtual Environment erstellt" -ForegroundColor Green
}

# Install dependencies
Write-Host "`nInstalliere Python Dependencies..." -ForegroundColor Yellow
& ".\venv\Scripts\activate.ps1"
& .\venv\Scripts\python.exe -m pip install --upgrade pip
& .\venv\Scripts\pip.exe install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
& .\venv\Scripts\pip.exe install -r requirements.txt
& .\venv\Scripts\python.exe -m spacy download de_core_news_sm
Write-Host "✓ Python Dependencies installiert" -ForegroundColor Green

# Docker images
Write-Host "`nLade Docker Images..." -ForegroundColor Yellow
& docker pull redis:7-alpine
& docker pull docker.n8n.io/n8nio/n8n
& docker pull postgres:16-alpine
& docker pull ollama/ollama:latest
& docker pull ghcr.io/open-webui/open-webui:main
Write-Host "✓ Docker Images geladen" -ForegroundColor Green

# Start services
Write-Host "`nStarte Docker Compose..." -ForegroundColor Yellow
& docker compose up -d

Start-Sleep -Seconds 30

Write-Host "`nServices sollten verfuegbar sein unter:" -ForegroundColor Green
Write-Host "  - N8N:         http://localhost:5678 (admin/changeme)" -ForegroundColor White
Write-Host "  - Open WebUI:  http://localhost:8080" -ForegroundColor White
Write-Host "  - Search API:  http://localhost:8001/docs" -ForegroundColor White
Write-Host "  - Redis:       localhost:6379" -ForegroundColor White

# Test document
$input = Read-Host "`nTest-Dokument erstellen? (j/n)"
if ($input -match '^[JjYy]') {
@"
Dies ist ein Test-Dokument für das Document Intelligence System.
Es enthält verschiedene Abschnitte und Informationen.

Features:
- Automatische Texterkennung
- DSGVO-konforme Verarbeitung
- Intelligente Verknüpfungen

Kontakt: test@example.com
Tel: +49 123 456789
"@ | Out-File -Encoding utf8 "data/test_document.txt"
    Write-Host "✓ Test-Dokument erstellt" -ForegroundColor Green
}

Write-Host "`n============================================================="
Write-Host "Installation abgeschlossen!" -ForegroundColor Green
Write-Host "============================================================="
Write-Host "`nNaechste Schritte:"
Write-Host "  1. .env Datei anpassen falls noetig"
Write-Host "  2. Dokumente in .\data ablegen"
Write-Host "  3. Open WebUI oeffnen fuer Suche"
Write-Host "`nNuetzliche Befehle:"
Write-Host "  docker compose logs -f    (Logs anzeigen)"
Write-Host "  docker compose down       (Services stoppen)"
Write-Host "  docker compose ps         (Status pruefen)"

Read-Host "`nDruecke Enter zum Beenden"
