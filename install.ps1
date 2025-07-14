# install.ps1 – Sichere PowerShell-Installation für das Document Intelligence System
# Version 2.0 - Mit umfassenden Fixes und Validierungen

#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$SkipGPU,
    [switch]$Force,
    [string]$LogFile = "install.log"
)

# Globale Variablen
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Global:InstallLog = @()

# Logging-Funktion
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry -ForegroundColor $(
        switch($Level) {
            "ERROR" { "Red" }
            "WARN" { "Yellow" }
            "SUCCESS" { "Green" }
            default { "White" }
        }
    )
    $Global:InstallLog += $logEntry
}

function Write-Header {
    param([string]$Title)
    Write-Host "`n$('='*60)" -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host $('='*60) -ForegroundColor Cyan
}

function Exit-WithError {
    param([string]$Message)
    Write-Log $Message "ERROR"
    
    # Speichere Log
    $Global:InstallLog | Out-File -FilePath $LogFile -Encoding utf8
    Write-Host "`nLog gespeichert in: $LogFile" -ForegroundColor Yellow
    
    Read-Host "Drücke Enter zum Beenden"
    Exit 1
}

function Test-Prerequisites {
    Write-Header "SYSTEM-VORAUSSETZUNGEN PRÜFEN"
    
    # 1. PowerShell Version
    $psVersion = $PSVersionTable.PSVersion
    Write-Log "PowerShell Version: $psVersion"
    if ($psVersion.Major -lt 5) {
        Exit-WithError "PowerShell 5.1+ erforderlich. Aktuelle Version: $psVersion"
    }
    
    # 2. Execution Policy
    $execPolicy = Get-ExecutionPolicy
    Write-Log "Execution Policy: $execPolicy"
    if ($execPolicy -eq "Restricted") {
        try {
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
            Write-Log "Execution Policy auf RemoteSigned gesetzt" "SUCCESS"
        } catch {
            Exit-WithError "Kann Execution Policy nicht ändern. Führe aus: Set-ExecutionPolicy RemoteSigned"
        }
    }
    
    # 3. Admin-Rechte
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole("Administrator")
    if (-not $isAdmin) {
        Exit-WithError "Administrator-Rechte erforderlich! Rechtsklick > Als Administrator ausführen"
    }
    Write-Log "Administrator-Rechte verfügbar" "SUCCESS"
    
    # 4. Verzeichnis-Validierung
    $currentPath = Get-Location
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
    
    # Verbotene Verzeichnisse
    $forbiddenPaths = @(
        "${env:SystemRoot}\System32",
        "${env:SystemRoot}",
        "${env:ProgramFiles}",
        "${env:ProgramFiles(x86)}"
    )
    
    foreach ($forbidden in $forbiddenPaths) {
        if ($currentPath.Path -like "$forbidden*") {
            Exit-WithError "Installation nicht aus '$($currentPath.Path)' möglich. Verwende Projektverzeichnis."
        }
    }
    
    # Wechsle ins Skriptverzeichnis
    if ($currentPath.Path -ne $scriptPath) {
        Set-Location $scriptPath
        Write-Log "Verzeichnis gewechselt zu: $scriptPath"
    }
    
    # 5. Prüfe ob docker-compose.yml existiert
    if (-not (Test-Path "docker-compose.yml")) {
        Exit-WithError "docker-compose.yml nicht gefunden. Bitte aus Projekt-Root ausführen."
    }
    
    Write-Log "Alle Voraussetzungen erfüllt" "SUCCESS"
}

function Test-Software {
    Write-Header "SOFTWARE-KOMPONENTEN PRÜFEN"
    
    # Python Check
    try {
        $pythonVersion = & python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Python gefunden: $pythonVersion" "SUCCESS"
            
            # Prüfe Python Version
            if ($pythonVersion -match "Python (\d+)\.(\d+)") {
                $major = [int]$matches[1]
                $minor = [int]$matches[2]
                if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
                    Exit-WithError "Python 3.10+ erforderlich. Gefunden: $pythonVersion"
                }
            }
        } else {
            Exit-WithError "Python nicht gefunden oder nicht im PATH"
        }
    } catch {
        Exit-WithError "Python-Check fehlgeschlagen: $($_.Exception.Message)"
    }
    
    # Docker Check
    try {
        $dockerVersion = & docker --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Docker gefunden: $dockerVersion" "SUCCESS"
        } else {
            Exit-WithError "Docker nicht gefunden. Installiere Docker Desktop von https://docker.com"
        }
    } catch {
        Exit-WithError "Docker-Check fehlgeschlagen. Ist Docker Desktop installiert?"
    }
    
    # Docker läuft?
    try {
        & docker ps | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Docker läuft" "SUCCESS"
        } else {
            Exit-WithError "Docker läuft nicht. Starte Docker Desktop."
        }
    } catch {
        Exit-WithError "Docker nicht erreichbar. Prüfe Docker Desktop."
    }
    
    # Docker Compose Check
    try {
        $composeVersion = & docker compose version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Docker Compose verfügbar: $composeVersion" "SUCCESS"
        } else {
            # Fallback: docker-compose
            $composeVersion = & docker-compose --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Log "Docker Compose (legacy) verfügbar: $composeVersion" "SUCCESS"
            } else {
                Exit-WithError "Docker Compose nicht verfügbar"
            }
        }
    } catch {
        Exit-WithError "Docker Compose Check fehlgeschlagen"
    }
}

function Test-GPU {
    Write-Header "GPU-UNTERSTÜTZUNG PRÜFEN"
    
    if ($SkipGPU) {
        Write-Log "GPU-Check übersprungen (--SkipGPU)" "WARN"
        return $false
    }
    
    try {
        # NVIDIA GPU Check
        $nvidiaOutput = & nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "NVIDIA GPU gefunden" "SUCCESS"
            Write-Log ($nvidiaOutput | Select-String "Driver Version" | Select-Object -First 1)
            
            # NVIDIA Container Toolkit Check
            try {
                & docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Log "NVIDIA Container Toolkit funktional" "SUCCESS"
                    return $true
                }
            } catch {
                Write-Log "NVIDIA Container Toolkit nicht verfügbar" "WARN"
            }
        }
    } catch {
        Write-Log "Keine NVIDIA GPU oder nvidia-smi nicht verfügbar" "WARN"
    }
    
    # AMD GPU Check (optional)
    try {
        $amdOutput = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*AMD*" -or $_.Name -like "*Radeon*" }
        if ($amdOutput) {
            Write-Log "AMD GPU gefunden: $($amdOutput.Name)" "WARN"
            Write-Log "AMD GPU-Unterstützung experimentell" "WARN"
        }
    } catch {
        # Ignoriere AMD GPU Fehler
    }
    
    Write-Log "Läuft im CPU-Modus" "WARN"
    return $false
}

function Initialize-Project {
    Write-Header "PROJEKT INITIALISIEREN"
    
    # Verzeichnisse erstellen
    $directories = @("data", "indices", "indices/json", "indices/markdown", "logs", "n8n/workflows")
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
            Write-Log "Verzeichnis erstellt: $dir"
        }
    }
    
    # .gitkeep Dateien
    @("data", "indices", "logs") | ForEach-Object {
        $gitkeepPath = Join-Path $_ ".gitkeep"
        if (-not (Test-Path $gitkeepPath)) {
            " " | Out-File -FilePath $gitkeepPath -Encoding ascii
        }
    }
    
    # .env Datei
    if (-not (Test-Path ".env")) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Log ".env aus .env.example erstellt" "SUCCESS"
        } else {
            # Erstelle minimale .env
            @"
# Document Intelligence System Configuration
DATA_PATH=./data
INDEX_PATH=./indices
LOG_PATH=./logs
REDIS_URL=redis://redis:6379
CORS_ALLOW_ORIGIN=http://localhost:8080,http://localhost:5678
USER_AGENT=DocumentIntelligenceSystem/1.0
CHROMA_TELEMETRY=false
TORCH_VERSION=2.1.0
"@ | Out-File -FilePath ".env" -Encoding utf8
            Write-Log ".env Datei erstellt" "SUCCESS"
        }
    } else {
        Write-Log ".env bereits vorhanden" "WARN"
    }
    
    Write-Log "Projekt initialisiert" "SUCCESS"
}

function Install-PythonDependencies {
    Write-Header "PYTHON-ABHÄNGIGKEITEN INSTALLIEREN"
    
    # Virtual Environment
    if (-not (Test-Path "venv")) {
        Write-Log "Erstelle Virtual Environment..."
        & python -m venv venv
        if ($LASTEXITCODE -ne 0) {
            Exit-WithError "Virtual Environment konnte nicht erstellt werden"
        }
        Write-Log "Virtual Environment erstellt" "SUCCESS"
    }
    
    # Aktiviere venv
    $venvActivate = ".\venv\Scripts\Activate.ps1"
    if (Test-Path $venvActivate) {
        & $venvActivate
        Write-Log "Virtual Environment aktiviert"
    } else {
        Exit-WithError "Virtual Environment Aktivierungsscript nicht gefunden"
    }
    
    # Upgrade pip
    Write-Log "Aktualisiere pip..."
    & .\venv\Scripts\python.exe -m pip install --upgrade pip
    
    # Installiere Requirements
    if (Test-Path "requirements.txt") {
        Write-Log "Installiere Python-Pakete..."
        
        # Spezielle PyTorch Installation für Windows
        Write-Log "Installiere PyTorch 2.1.0..."
        & .\venv\Scripts\pip.exe install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
        
        # Restliche Requirements
        & .\venv\Scripts\pip.exe install -r requirements.txt
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Einige Pakete konnten nicht installiert werden" "WARN"
        } else {
            Write-Log "Python-Pakete installiert" "SUCCESS"
        }
    }
    
    # Spacy Modell
    Write-Log "Lade Spacy Deutsch-Modell..."
    & .\venv\Scripts\python.exe -m spacy download de_core_news_sm
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Spacy Modell installiert" "SUCCESS"
    } else {
        Write-Log "Spacy Modell Installation fehlgeschlagen" "WARN"
    }
}

function Start-DockerServices {
    param([bool]$HasGPU)
    
    Write-Header "DOCKER SERVICES STARTEN"
    
    # Pull Images first
    Write-Log "Lade Docker Images..."
    
    $images = @(
        "redis:7-alpine",
        "postgres:16-alpine", 
        "docker.n8n.io/n8nio/n8n",
        "ghcr.io/open-webui/open-webui:main"
    )
    
    if (-not $SkipGPU -and $HasGPU) {
        $images += "ollama/ollama:latest"
    }
    
    foreach ($image in $images) {
        Write-Log "Lade $image..."
        & docker pull $image
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Warnung: $image konnte nicht geladen werden" "WARN"
        }
    }
    
    # Wähle compose Datei
    $composeFile = "docker-compose.yml"
    if ($SkipGPU -or -not $HasGPU) {
        if (Test-Path "docker-compose-cpu.yml") {
            $composeFile = "docker-compose-cpu.yml"
            Write-Log "Verwende CPU-only Konfiguration"
        }
    }
    
    # Stoppe existierende Services
    Write-Log "Stoppe existierende Services..."
    & docker compose -f $composeFile down 2>$null
    
    # Starte Services
    Write-Log "Starte Docker Services..."
    & docker compose -f $composeFile up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Docker Services gestartet" "SUCCESS"
    } else {
        Exit-WithError "Docker Services konnten nicht gestartet werden"
    }
    
    # Warte auf Services
    Write-Log "Warte auf Service-Initialisierung..."
    Start-Sleep -Seconds 30
}

function Test-Services {
    Write-Header "SERVICE-VERFÜGBARKEIT TESTEN"
    
    $services = @(
        @{ Name = "Redis"; URL = "http://localhost:6379"; Expected = $null },
        @{ Name = "N8N"; URL = "http://localhost:5678"; Expected = 200 },
        @{ Name = "Search API"; URL = "http://localhost:8001/health"; Expected = 200 },
        @{ Name = "Open WebUI"; URL = "http://localhost:8080"; Expected = 200 }
    )
    
    $allHealthy = $true
    
    foreach ($service in $services) {
        try {
            if ($service.Name -eq "Redis") {
                # Redis spezielle Prüfung
                $redisTest = & docker exec document-intelligence-system-redis-1 redis-cli ping 2>&1
                if ($redisTest -match "PONG") {
                    Write-Log "$($service.Name): Verfügbar" "SUCCESS"
                } else {
                    Write-Log "$($service.Name): Nicht erreichbar" "WARN"
                    $allHealthy = $false
                }
            } else {
                # HTTP Services
                $response = Invoke-WebRequest -Uri $service.URL -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
                if ($response.StatusCode -eq $service.Expected) {
                    Write-Log "$($service.Name): Verfügbar ($($service.URL))" "SUCCESS"
                } else {
                    Write-Log "$($service.Name): Unerwarteter Status $($response.StatusCode)" "WARN"
                    $allHealthy = $false
                }
            }
        } catch {
            Write-Log "$($service.Name): Nicht erreichbar - $($_.Exception.Message)" "WARN"
            $allHealthy = $false
        }
    }
    
    if ($allHealthy) {
        Write-Log "Alle Services verfügbar" "SUCCESS"
    } else {
        Write-Log "Einige Services nicht verfügbar - prüfe Logs mit: docker compose logs" "WARN"
    }
    
    return $allHealthy
}

function Install-OllamaModels {
    Write-Header "OLLAMA MODELLE INSTALLIEREN (OPTIONAL)"
    
    # Prüfe ob Ollama läuft
    try {
        $ollamaTest = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 5
        if ($ollamaTest.StatusCode -eq 200) {
            Write-Log "Ollama verfügbar"
            
            $models = @("mistral", "llama3")
            foreach ($model in $models) {
                Write-Log "Installiere Modell: $model..."
                & docker exec document-intelligence-system-ollama-1 ollama pull $model
                if ($LASTEXITCODE -eq 0) {
                    Write-Log "Modell $model installiert" "SUCCESS"
                } else {
                    Write-Log "Modell $model Installation fehlgeschlagen" "WARN"
                }
            }
        }
    } catch {
        Write-Log "Ollama nicht verfügbar - überspringe Modell-Installation" "WARN"
    }
}

function Create-TestDocument {
    Write-Header "TEST-DOKUMENT ERSTELLEN"
    
    $testContent = @"
Document Intelligence System Test

Dies ist ein Test-Dokument für das Document Intelligence System.
Erstellt am: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

Features:
- Automatische OCR-Verarbeitung
- DSGVO-konforme Indexierung  
- Intelligente Suche
- KI-gestützte Verknüpfungen

Test-Keywords: installation, windows, powershell, test
Kontakt: test@example.com
Telefon: +49 123 456789

Dieses Dokument wird automatisch vom Watchdog erkannt und verarbeitet.
"@
    
    $testPath = "data\test_installation.txt"
    $testContent | Out-File -FilePath $testPath -Encoding utf8
    Write-Log "Test-Dokument erstellt: $testPath" "SUCCESS"
    
    # Warte kurz auf Verarbeitung
    Write-Log "Warte 15 Sekunden auf automatische Verarbeitung..."
    Start-Sleep -Seconds 15
    
    # Teste Suche
    try {
        $searchBody = @{
            query = "installation test"
            limit = 5
        } | ConvertTo-Json
        
        $searchResult = Invoke-RestMethod -Uri "http://localhost:8001/search" -Method Post -Body $searchBody -ContentType "application/json" -TimeoutSec 10
        
        if ($searchResult -and $searchResult.Count -gt 0) {
            Write-Log "Test-Dokument erfolgreich indiziert und gefunden!" "SUCCESS"
            Write-Log "Gefundene Ergebnisse: $($searchResult.Count)"
        } else {
            Write-Log "Test-Dokument noch nicht indiziert - das ist normal" "WARN"
        }
    } catch {
        Write-Log "Suche-Test nicht möglich: $($_.Exception.Message)" "WARN"
    }
}

function Show-Summary {
    Write-Header "INSTALLATION ABGESCHLOSSEN"
    
    Write-Host @"

🎉 Document Intelligence System erfolgreich installiert!

📍 SERVICE-ADRESSEN:
   • N8N Workflow:    http://localhost:5678 (admin/changeme)
   • Open WebUI:      http://localhost:8080
   • Search API:      http://localhost:8001/docs  
   • Redis:           localhost:6379

📋 NÄCHSTE SCHRITTE:
   1. Passe .env Konfiguration nach Bedarf an
   2. Lege Dokumente in .\data Ordner
   3. Öffne Open WebUI für die Suche
   4. Konfiguriere N8N Workflows

🔧 NÜTZLICHE BEFEHLE:
   • docker compose logs -f          (Live-Logs anzeigen)
   • docker compose ps               (Service-Status)  
   • docker compose down             (Services stoppen)
   • python check_fixes.py           (System-Check)

📁 WICHTIGE VERZEICHNISSE:
   • .\data\         → Dokumente hier ablegen
   • .\indices\      → Generierte Indizes
   • .\logs\         → System-Logs

"@ -ForegroundColor Green

    # Speichere Installations-Log
    $Global:InstallLog | Out-File -FilePath $LogFile -Encoding utf8
    Write-Host "📋 Installations-Log gespeichert: $LogFile" -ForegroundColor Yellow
}

# MAIN INSTALLATION ROUTINE
function Main {
    try {
        Write-Header "DOCUMENT INTELLIGENCE SYSTEM - INSTALLATION"
        Write-Log "Installation gestartet um $(Get-Date)"
        Write-Log "PowerShell Version: $($PSVersionTable.PSVersion)"
        Write-Log "System: $($env:COMPUTERNAME) - $($env:USERNAME)"
        
        # Installation Steps
        Test-Prerequisites
        Test-Software  
        $hasGPU = Test-GPU
        Initialize-Project
        Install-PythonDependencies
        Start-DockerServices -HasGPU $hasGPU
        $servicesHealthy = Test-Services
        
        if (-not $SkipGPU -and $hasGPU) {
            Install-OllamaModels
        }
        
        Create-TestDocument
        Show-Summary
        
        Write-Log "Installation erfolgreich abgeschlossen!" "SUCCESS"
        
    } catch {
        Write-Log "Installation fehlgeschlagen: $($_.Exception.Message)" "ERROR"
        Write-Log "Stack Trace: $($_.ScriptStackTrace)" "ERROR"
        Exit-WithError "Schwerwiegender Fehler aufgetreten"
    }
}

# Script Entry Point
if ($MyInvocation.InvocationName -ne '.') {
    Main
}
