# install.ps1 – Sichere PowerShell-Installation für das Document Intelligence System

Write-Host "`n=== Document Intelligence System Installation (PowerShell) ===" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan

# Funktion für Exit bei Fehler
function Exit-WithError {
    param($msg)
    Write-Host "`nFEHLER: $msg" -ForegroundColor Red
    Pause
    Exit 1
}

# Verzeichnis validieren
$current = (Get-Location).Path
$scriptPath = $MyInvocation.MyCommand.Path | Split-Path -Parent

# Liste verbotener Ordner
$bannedDirs = @(
    "C:\Windows\System32",
    "C:\Windows",
    "C:\Program Files",
    "C:\Program Files (x86)"
)

if ($bannedDirs -contains $current) {
    Exit-WithError "Das Skript darf nicht aus '$current' ausgeführt werden. Bitte aus dem Projektverzeichnis starten."
}

# In Skriptverzeichnis wechseln, falls nötig
if ($current -ne $scriptPath) {
    Write-Host "Wechsle ins Skriptverzeichnis: $scriptPath" -ForegroundColor Yellow
    Set-Location $scriptPath
}

# Admin-Check
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole("Administrator")) {
    Exit-WithError "Bitte führe dieses Skript als Administrator aus (Rechtsklick > Als Administrator ausführen)"
}

# Verzeichnisstruktur
Write-Host "`n[1/7] Erstelle Verzeichnisse..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "data", "indices/json", "indices/markdown", "logs", "n8n/workflows" | Out-Null
" " | Set-Content "data/.gitkeep"
" " | Set-Content "indices/.gitkeep"
" " | Set-Content "logs/.gitkeep"
Write-Host "✓ Verzeichnisse erstellt" -ForegroundColor Green

# .env Setup
Write-Host "`n[2/7] Setup .env Datei..." -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✓ .env Datei erstellt" -ForegroundColor Green
    Write-Host "⚠ Bitte .env Datei nach Bedarf anpassen!" -ForegroundColor Yellow
} else {
    Write-Host "⚠ .env existiert bereits, wird nicht überschrieben" -ForegroundColor Yellow
}

# Python Check
Write-Host "`n[3/7] Überprüfe Python..." -ForegroundColor Cyan
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Exit-WithError "Python ist nicht installiert oder nicht im PATH"
}
Write-Host "✓ Python gefunden: $($python.Source)" -ForegroundColor Green

# Docker Check
Write-Host "`n[4/7] Überprüfe Docker..." -ForegroundColor Cyan
$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    Exit-WithError "Docker ist nicht installiert oder nicht im PATH"
}
Write-Host "✓ Docker gefunden: $($docker.Source)" -ForegroundColor Green

# Docker läuft?
docker ps > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Exit-WithError "Docker läuft nicht. Bitte Docker Desktop starten."
}
Write-Host "✓ Docker läuft" -ForegroundColor Green

# Python venv + Abhängigkeiten
Write-Host "`n[5/7] Installiere Python-Abhängigkeiten..." -ForegroundColor Cyan
if (-not (Test-Path "venv")) {
    python -m venv venv
}
& ".\venv\Scripts\activate.ps1"
pip install --upgrade pip
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Host "⚠ requirements.txt fehlt – wird übersprungen" -ForegroundColor Yellow
}
python -m spacy download de_core_news_sm
Write-Host "✓ Python-Abhängigkeiten installiert" -ForegroundColor Green

# Docker Images einzeln ziehen
Write-Host "`n[6/7] Docker-Images pullen..." -ForegroundColor Cyan
docker pull redis:7-alpine
docker pull postgres:16-alpine
docker pull docker.n8n.io/n8nio/n8n
docker pull ghcr.io/open-webui/open-webui:main
docker pull ollama/ollama:latest
Write-Host "✓ Docker-Images geladen" -ForegroundColor Green

# Compose starten
Write-Host "`n[7/7] Starte Docker Compose..." -ForegroundColor Cyan
docker compose up -d

Start-Sleep -Seconds 10

# Health-Checks
function Check-Service {
    param($url, $name)
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
        if ($resp.StatusCode -eq 200) {
            Write-Host "✓ $name läuft unter $url" -ForegroundColor Green
        } else {
            Write-Host "⚠ $name antwortet nicht korrekt: $url" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠ $name nicht erreichbar unter $url" -ForegroundColor Yellow
    }
}

Check-Service "http://localhost:5678" "N8N"
Check-Service "http://localhost:8001" "Search API"
Check-Service "http://localhost:8080" "Open WebUI"

# Optionales Test-Dokument
$input = Read-Host "Test-Dokument erstellen? (j/n)"
if ($input -match '^[JjYy]') {
@"
Dies ist ein Test-Dokument für das Document Intelligence System.
Es enthält PII und strukturierte Abschnitte zur Indexierung.
Siehe auch: Datenschutzrichtlinie
Kontakt: max@example.com
"@ | Out-File -Encoding utf8 "data/test_document.txt"
    Write-Host "✓ Test-Dokument erstellt in data/test_document.txt" -ForegroundColor Green
}

Write-Host "`n============================================================="
Write-Host "Installation abgeschlossen! Viel Erfolg." -ForegroundColor Green
Write-Host "=============================================================`n"
