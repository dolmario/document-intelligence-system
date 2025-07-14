#!/bin/bash

# Document Intelligence System - Linux/Mac/WSL Installation
# Funktioniert auf: Ubuntu, Debian, CentOS, RHEL, macOS, WSL

set -e  # Stoppe bei Fehlern

echo "=== Document Intelligence System Installation (Linux/Mac/WSL) ==="
echo "================================================================="

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging-Funktionen
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Fehler-Handler
error_exit() {
    log_error "$1"
    echo ""
    echo "Installation fehlgeschlagen!"
    echo "Logs siehe oben für Details."
    exit 1
}

# System-Erkennung
detect_system() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS_NAME=$NAME
            OS_VERSION=$VERSION_ID
        else
            OS_NAME="Linux"
            OS_VERSION="Unknown"
        fi
        
        # WSL Erkennung
        if grep -q microsoft /proc/version 2>/dev/null; then
            OS_NAME="WSL ($OS_NAME)"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_NAME="macOS"
        OS_VERSION=$(sw_vers -productVersion)
    else
        OS_NAME="Unknown"
        OS_VERSION="Unknown"
    fi
    
    log_info "System erkannt: $OS_NAME $OS_VERSION"
}

# Sudo/Root Check
check_permissions() {
    log_info "Prüfe Berechtigungen..."
    
    if [[ $EUID -eq 0 ]]; then
        log_warn "Läuft als Root - das ist OK aber nicht nötig"
        SUDO_CMD=""
    else
        # Prüfe ob sudo verfügbar ist
        if command -v sudo >/dev/null 2>&1; then
            SUDO_CMD="sudo"
            log_info "Verwende sudo für System-Befehle"
        else
            log_warn "Sudo nicht verfügbar - einige Schritte könnten fehlschlagen"
            SUDO_CMD=""
        fi
    fi
}

# Projektverzeichnis prüfen
check_project_directory() {
    log_info "Prüfe Projektverzeichnis..."
    
    # Wechsel ins Skriptverzeichnis
    cd "$(dirname "$0")"
    
    if [ ! -f "docker-compose.yml" ]; then
        error_exit "docker-compose.yml nicht gefunden! Bitte aus Projekt-Root ausführen."
    fi
    
    log_success "Richtiges Projektverzeichnis gefunden: $(pwd)"
}

# System-Dependencies installieren
install_system_deps() {
    log_info "Installiere System-Dependencies..."
    
    if [[ "$OS_NAME" == *"Ubuntu"* ]] || [[ "$OS_NAME" == *"Debian"* ]]; then
        # Ubuntu/Debian
        $SUDO_CMD apt-get update
        $SUDO_CMD apt-get install -y curl wget git python3 python3-pip python3-venv
        
        # Docker installieren falls nicht vorhanden
        if ! command -v docker >/dev/null 2>&1; then
            log_info "Installiere Docker..."
            curl -fsSL https://get.docker.com -o get-docker.sh
            $SUDO_CMD sh get-docker.sh
            $SUDO_CMD usermod -aG docker $USER
            rm get-docker.sh
            log_warn "Docker installiert - bitte neu einloggen oder 'newgrp docker' ausführen"
        fi
        
    elif [[ "$OS_NAME" == *"CentOS"* ]] || [[ "$OS_NAME" == *"Red Hat"* ]] || [[ "$OS_NAME" == *"Rocky"* ]]; then
        # RHEL/CentOS/Rocky
        $SUDO_CMD yum update -y
        $SUDO_CMD yum install -y curl wget git python3 python3-pip
        
        # Docker installieren
        if ! command -v docker >/dev/null 2>&1; then
            log_info "Installiere Docker..."
            curl -fsSL https://get.docker.com -o get-docker.sh
            $SUDO_CMD sh get-docker.sh
            $SUDO_CMD systemctl start docker
            $SUDO_CMD systemctl enable docker
            $SUDO_CMD usermod -aG docker $USER
            rm get-docker.sh
        fi
        
    elif [[ "$OS_NAME" == *"macOS"* ]]; then
        # macOS - verwende Homebrew
        if ! command -v brew >/dev/null 2>&1; then
            log_info "Installiere Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Dependencies via Homebrew
        brew install python3 git curl wget
        
        # Docker Desktop Check
        if ! command -v docker >/dev/null 2>&1; then
            log_warn "Docker nicht gefunden. Bitte Docker Desktop von https://docker.com installieren"
            read -p "Docker Desktop installiert? (j/n): " docker_installed
            if [[ "$docker_installed" != "j" ]] && [[ "$docker_installed" != "y" ]]; then
                error_exit "Docker ist erforderlich"
            fi
        fi
        
    elif [[ "$OS_NAME" == *"WSL"* ]]; then
        # WSL - ähnlich Ubuntu aber Docker Desktop verwenden
        $SUDO_CMD apt-get update
        $SUDO_CMD apt-get install -y curl wget git python3 python3-pip python3-venv
        
        log_info "WSL erkannt - stelle sicher dass Docker Desktop läuft"
        
    else
        log_warn "Unbekanntes System - versuche generische Installation"
    fi
}

# Software-Checks
check_software() {
    log_info "Prüfe Software-Voraussetzungen..."
    
    # Python Check
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_success "Python gefunden: $PYTHON_VERSION"
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_VERSION=$(python --version | cut -d' ' -f2)
        log_success "Python gefunden: $PYTHON_VERSION"
        PYTHON_CMD="python"
    else
        error_exit "Python nicht gefunden. Bitte installieren."
    fi
    
    # Docker Check
    if command -v docker >/dev/null 2>&1; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        log_success "Docker gefunden: $DOCKER_VERSION"
    else
        error_exit "Docker nicht gefunden. Bitte installieren: https://docs.docker.com/get-docker/"
    fi
    
    # Docker läuft?
    if docker ps >/dev/null 2>&1; then
        log_success "Docker läuft"
    else
        error_exit "Docker läuft nicht. Bitte starten: sudo systemctl start docker"
    fi
    
    # Docker Compose Check
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_VERSION=$(docker compose version --short)
        log_success "Docker Compose gefunden: $COMPOSE_VERSION"
    elif docker-compose --version >/dev/null 2>&1; then
        log_success "Docker Compose (legacy) gefunden"
        # Alias für einheitliche Nutzung
        alias docker-compose='docker compose'
    else
        error_exit "Docker Compose nicht gefunden"
    fi
}

# GPU-Erkennung (NVIDIA)
check_gpu() {
    log_info "Prüfe GPU-Unterstützung..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_success "NVIDIA GPU gefunden"
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1
        
        # NVIDIA Container Toolkit Check
        if docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
            log_success "NVIDIA Container Toolkit funktional"
            GPU_SUPPORT=true
        else
            log_warn "NVIDIA Container Toolkit nicht installiert"
            log_info "Installiere mit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            GPU_SUPPORT=false
        fi
    else
        log_warn "Keine NVIDIA GPU gefunden - Ollama läuft auf CPU"
        GPU_SUPPORT=false
    fi
}

# Projektverzeichnisse erstellen
create_directories() {
    log_info "Erstelle Verzeichnisstruktur..."
    
    mkdir -p data indices/{json,markdown} logs n8n/workflows
    
    # .gitkeep Dateien
    touch data/.gitkeep indices/.gitkeep logs/.gitkeep
    
    log_success "Verzeichnisse erstellt"
}

# Environment Setup
setup_environment() {
    log_info "Setup Umgebungskonfiguration..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            log_success ".env aus .env.example erstellt"
        else
            # Erstelle minimale .env
            cat > .env << 'EOF'
# Document Intelligence System Configuration
DATA_PATH=./data
INDEX_PATH=./indices
LOG_PATH=./logs
REDIS_URL=redis://redis:6379
CORS_ALLOW_ORIGIN=http://localhost:8080,http://localhost:5678
USER_AGENT=DocumentIntelligenceSystem/1.0
CHROMA_TELEMETRY=false
TORCH_VERSION=2.1.0
EOF
            log_success ".env Datei erstellt"
        fi
        log_warn "Du kannst .env später anpassen falls nötig"
    else
        log_success ".env bereits vorhanden"
    fi
}

# Python Dependencies
install_python_deps() {
    log_info "Installiere Python Dependencies..."
    log_info "Das kann einige Minuten dauern..."
    
    # Virtual Environment
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        log_success "Virtual Environment erstellt"
    else
        log_success "Virtual Environment bereits vorhanden"
    fi
    
    # Aktiviere venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # PyTorch zuerst (wichtig für Kompatibilität)
    log_info "Installiere PyTorch 2.1.0 (kompatible Version)..."
    
    # Unterscheide zwischen CPU und GPU Installation
    if [ "$GPU_SUPPORT" = true ]; then
        pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
    fi
    
    # Requirements installieren
    if [ -f requirements.txt ]; then
        log_info "Installiere weitere Dependencies..."
        pip install -r requirements.txt || log_warn "Einige Pakete konnten nicht installiert werden"
    else
        log_warn "requirements.txt nicht gefunden - installiere Basis-Pakete"
        pip install redis fastapi uvicorn spacy pytesseract Pillow pdf2image
    fi
    
    # Spacy Modell
    log_info "Lade Spacy Deutsch-Modell..."
    python -m spacy download de_core_news_sm || log_warn "Spacy Modell Installation fehlgeschlagen"
    
    log_success "Python Dependencies installiert"
}

# Docker Images laden
pull_docker_images() {
    log_info "Lade Docker Images..."
    log_info "Das kann beim ersten Mal sehr lange dauern!"
    
    images=(
        "redis:7-alpine"
        "postgres:16-alpine"
        "docker.n8n.io/n8nio/n8n"
        "ghcr.io/open-webui/open-webui:main"
    )
    
    # Ollama nur bei GPU-Support
    if [ "$GPU_SUPPORT" = true ]; then
        images+=("ollama/ollama:latest")
    fi
    
    for image in "${images[@]}"; do
        log_info "Lade $image..."
        docker pull "$image" || log_warn "$image konnte nicht geladen werden"
    done
    
    log_success "Docker Images geladen"
}

# Services starten
start_services() {
    log_info "Starte Docker Services..."
    
    # Wähle compose Datei
    COMPOSE_FILE="docker-compose.yml"
    if [ "$GPU_SUPPORT" = false ] && [ -f "docker-compose-cpu.yml" ]; then
        COMPOSE_FILE="docker-compose-cpu.yml"
        log_info "Verwende CPU-only Konfiguration"
    fi
    
    # Stoppe existierende Services
    docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
    
    # Starte Services
    docker compose -f "$COMPOSE_FILE" up -d
    
    if [ $? -eq 0 ]; then
        log_success "Docker Services gestartet"
    else
        error_exit "Docker Services konnten nicht gestartet werden"
    fi
    
    # Warte auf Services
    log_info "Warte 30 Sekunden auf Service-Initialisierung..."
    sleep 30
}

# Service Health Checks
test_services() {
    log_info "Prüfe Service-Verfügbarkeit..."
    
    # Einfache Container-Checks
    if docker ps | grep -q redis; then
        log_success "Redis läuft"
    else
        log_warn "Redis Problem"
    fi
    
    if docker ps | grep -q n8n; then
        log_success "N8N läuft"
    else
        log_warn "N8N Problem"
    fi
    
    if docker ps | grep -q search; then
        log_success "Search API läuft"
    else
        log_warn "Search API Problem"
    fi
    
    if docker ps | grep -q webui; then
        log_success "Open WebUI läuft"
    else
        log_warn "WebUI Problem"
    fi
    
    # HTTP-Checks (optional)
    if command -v curl >/dev/null 2>&1; then
        log_info "Teste HTTP Endpoints..."
        
        if curl -s http://localhost:5678 >/dev/null; then
            log_success "N8N erreichbar: http://localhost:5678"
        else
            log_warn "N8N nicht erreichbar"
        fi
        
        if curl -s http://localhost:8080 >/dev/null; then
            log_success "Open WebUI erreichbar: http://localhost:8080"
        else
            log_warn "Open WebUI nicht erreichbar"
        fi
        
        if curl -s http://localhost:8001 >/dev/null; then
            log_success "Search API erreichbar: http://localhost:8001"
        else
            log_warn "Search API nicht erreichbar"
        fi
    fi
}

# Ollama Modelle installieren (optional)
install_ollama_models() {
    if [ "$GPU_SUPPORT" = false ]; then
        log_info "Überspringe Ollama Modelle (kein GPU)"
        return
    fi
    
    log_info "Installiere Ollama Modelle (optional)..."
    
    # Warte bis Ollama bereit ist
    local retries=0
    while [ $retries -lt 30 ]; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            break
        fi
        sleep 2
        ((retries++))
    done
    
    if [ $retries -eq 30 ]; then
        log_warn "Ollama nicht verfügbar - überspringe Modell-Installation"
        return
    fi
    
    models=("mistral" "llama3")
    
    for model in "${models[@]}"; do
        log_info "Installiere Modell: $model..."
        docker exec document-intelligence-system-ollama-1 ollama pull "$model" || log_warn "Modell $model Installation fehlgeschlagen"
    done
}

# Test-Dokument erstellen
create_test_document() {
    log_info "Test-Dokument erstellen?"
    read -p "Test-Dokument erstellen? (j/n): " create_test
    
    if [[ "$create_test" =~ ^[JjYy]$ ]]; then
        cat > data/test_installation.txt << 'EOF'
Document Intelligence System Test

Dies ist ein Test-Dokument für das Document Intelligence System.
Erstellt am: $(date)

Features:
- Automatische OCR-Verarbeitung
- DSGVO-konforme Indexierung
- Intelligente Suche
- KI-gestützte Verknüpfungen

Test-Keywords: installation, linux, bash, test
Kontakt: test@example.com
Telefon: +49 123 456789

Dieses Dokument wird automatisch vom Watchdog erkannt und verarbeitet.
EOF
        log_success "Test-Dokument erstellt: data/test_installation.txt"
    fi
}

# Zusammenfassung anzeigen
show_summary() {
    echo ""
    echo "================================================================"
    echo -e "${GREEN}✅ INSTALLATION ERFOLGREICH ABGESCHLOSSEN!${NC}"
    echo "================================================================"
    echo ""
    echo -e "${BLUE}🌐 SERVICE-ADRESSEN:${NC}"
    echo "  • N8N Workflow:    http://localhost:5678 (admin/changeme)"
    echo "  • Open WebUI:      http://localhost:8080"
    echo "  • Search API:      http://localhost:8001/docs"
    echo "  • Redis:           localhost:6379"
    echo ""
    echo -e "${BLUE}📋 NÄCHSTE SCHRITTE:${NC}"
    echo "  1. Passe .env Datei nach Bedarf an"
    echo "  2. Lege Dokumente in ./data Ordner"
    echo "  3. Öffne http://localhost:8080 für die Suche"
    echo "  4. Konfiguriere N8N Workflows unter http://localhost:5678"
    echo ""
    echo -e "${BLUE}🔧 NÜTZLICHE BEFEHLE:${NC}"
    echo "  • docker compose logs -f          (Live-Logs anzeigen)"
    echo "  • docker compose ps               (Service-Status)"
    echo "  • docker compose down             (Services stoppen)"
    echo "  • docker compose restart          (Services neu starten)"
    echo ""
    echo -e "${BLUE}📁 WICHTIGE VERZEICHNISSE:${NC}"
    echo "  • ./data/         → Dokumente hier ablegen"
    echo "  • ./indices/      → Generierte Indizes"
    echo "  • ./logs/         → System-Logs"
    echo ""
    echo -e "${BLUE}💡 PROBLEME?${NC}"
    echo "  • Logs prüfen: docker compose logs"
    echo "  • Services neu starten: docker compose restart"
    echo "  • Alles stoppen: docker compose down"
    echo ""
}

# MAIN INSTALLATION ROUTINE
main() {
    echo ""
    log_info "Installation gestartet um $(date)"
    
    # Installation Steps
    detect_system
    check_permissions
    check_project_directory
    
    # Optional: System-Dependencies installieren
    read -p "System-Dependencies automatisch installieren? (j/n): " install_deps
    if [[ "$install_deps" =~ ^[JjYy]$ ]]; then
        install_system_deps
    fi
    
    check_software
    check_gpu
    create_directories
    setup_environment
    install_python_deps
    pull_docker_images
    start_services
    test_services
    
    if [ "$GPU_SUPPORT" = true ]; then
        install_ollama_models
    fi
    
    create_test_document
    show_summary
    
    log_success "Installation erfolgreich abgeschlossen!"
}

# Script Entry Point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
