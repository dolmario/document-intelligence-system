# install.ps1 - Windows Installation Script

Write-Host "=== Document Intelligence System Installation (Windows) ===" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# Funktion für Fehlerbehandlung
function Exit-WithError {
    param($Message)
    Write-Host "FEHLER: $Message" -ForegroundColor Red
    exit 1
}

# Funktion für Success
function Write-Success {
    param($Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

# Funktion für Warnings
function Write-Warning {
    param($Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

# Admin Check
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Exit-WithError "Bitte als Administrator ausführen!"
}

# System Checks
Write-Host "`nChecking System Requirements
