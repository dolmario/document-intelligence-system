#!/bin/bash

# Document Intelligence System - Installation Script
# Supports: Ubuntu, Debian, CentOS, RHEL, macOS, WSL

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

error_exit() {
    log_error "$1"
    exit 1
}

# System Detection
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
    
    log_info "System detected: $OS_NAME $OS_VERSION"
}

# Check permissions
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        log_warn "Running as root"
        SUDO_CMD=""
    else
        if command -v sudo >/dev/null 2>&1; then
            SUDO_CMD="sudo"
        else
            SUDO_CMD=""
        fi
    fi
}

# Check project directory
check_project_directory() {
    cd "$(dirname "$0")"
    
    if [ ! -f "docker-compose.yml" ]; then
        error_exit "docker-compose.yml not found! Please run from project root."
    fi
    
    log_success "Project directory: $(pwd)"
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    if [[ "$OS_NAME" == *"Ubuntu"* ]] || [[ "$OS_NAME" == *"Debian"* ]]; then
        $SUDO_CMD apt-get update
        $SUDO_CMD apt-get install -y curl wget git python3 python3-pip python3-venv
        
        if ! command -v docker >/dev/null 2>&1; then
            log_info "Installing Docker..."
            curl -fsSL https://get.docker.com -o get-docker.sh
            $SUDO_CMD sh get-docker.sh
            $SUDO_CMD usermod -aG docker $USER
            rm get-docker.sh
        fi
        
    elif [[ "$OS_NAME" == *"CentOS"* ]] || [[ "$OS_NAME" == *"Red Hat"* ]]; then
        $SUDO_CMD yum update -y
        $SUDO_CMD yum install -y curl wget git python3 python3-pip
        
        if ! command -v docker >/dev/null 2>&1; then
            curl -fsSL https://get.docker.com -o get-docker.sh
            $SUDO_CMD sh get-docker.sh
            $SUDO_CMD systemctl start docker
            $SUDO_CMD systemctl enable docker
            $SUDO_CMD usermod -aG docker $USER
            rm get-docker.sh
        fi
        
    elif [[ "$OS_NAME" == *"macOS"* ]]; then
        if ! command -v brew >/dev/null 2>&1; then
            log_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        brew install python3 git curl wget
        
        if ! command -v docker >/dev/null 2>&1; then
            log_warn "Docker Desktop required. Please install from https://docker.com"
            read -p "Docker Desktop installed? (y/n): " docker_installed
            if [[ "$docker_installed" != "y" ]]; then
                error_exit "Docker is required"
            fi
        fi
    fi
}

# Check software requirements
check_software() {
    log_info "Checking software requirements..."
    
    # Python
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_success "Python: $PYTHON_VERSION"
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_VERSION=$(python --version | cut -d' ' -f2)
        log_success "Python: $PYTHON_VERSION"
        PYTHON_CMD="python"
    else
        error_exit "Python not found"
    fi
    
    # Docker
    if command -v docker >/dev/null 2>&1; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        log_success "Docker: $DOCKER_VERSION"
    else
        error_exit "Docker not found"
    fi
    
    # Docker running
    if docker ps >/dev/null 2>&1; then
        log_success "Docker is running"
    else
        error_exit "Docker is not running"
    fi
    
    # Docker Compose
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_VERSION=$(docker compose version --short)
        log_success "Docker Compose: $COMPOSE_VERSION"
    else
        error_exit "Docker Compose not found"
    fi
}

# GPU Detection
check_gpu() {
    log_info "Checking GPU support..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1
        
        if docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
            log_success "NVIDIA Container Toolkit functional"
            GPU_SUPPORT=true
        else
            log_warn "NVIDIA Container Toolkit not installed"
            GPU_SUPPORT=false
        fi
    else
        log_warn "No NVIDIA GPU found - using CPU mode"
        GPU_SUPPORT=false
    fi
}

# Create directories
create_directories() {
    log_info "Creating directory structure..."
    
    mkdir -p data indices/{json,markdown} logs n8n/workflows
    touch data/.gitkeep indices/.gitkeep logs/.gitkeep
    
    log_success "Directories created"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            log_success ".env created from .env.example"
        else
            cat > .env << 'EOF'
DATA_PATH=./data
INDEX_PATH=./indices
LOG_PATH=./logs
REDIS_URL=redis://redis:6379
CORS_ALLOW_ORIGIN=http://localhost:8080,http://localhost:5678
USER_AGENT=DocumentIntelligenceSystem/1.0
CHROMA_TELEMETRY=false
TORCH_VERSION=2.1.0
POSTGRES_PASSWORD=docintell123
N8N_BASIC_AUTH_PASSWORD=changeme
WEBUI_SECRET_KEY=change-me-in-production
EOF
            log_success ".env file created"
        fi
    else
        log_success ".env already exists"
    fi
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        log_success "Virtual environment created"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    
    if [ "$GPU_SUPPORT" = true ]; then
        pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
    fi
    
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt
    fi
    
    python -m spacy download de_core_news_sm || log_warn "Spacy model installation failed"
    
    log_success "Python dependencies installed"
}

# Build and start services
start_services() {
    log_info "Building and starting services..."
    
    docker compose down -v >/dev/null 2>&1 || true
    
    log_info "Building containers..."
    docker compose build --no-cache
    
    log_info "Starting core services..."
    docker compose up -d redis postgres
    sleep 15
    
    log_info "Starting agents..."
    docker compose up -d watchdog ocr_agent indexer
    sleep 10
    
    log_info "Starting API services..."
    docker compose up -d search_api n8n
    sleep 10
    
    log_info "Starting Ollama and WebUI..."
    docker compose up -d ollama open-webui
    
    log_success "All services started"
}

# Test services
test_services() {
    log_info "Testing services..."
    
    sleep 20
    
    # Check containers
    if docker compose ps | grep -q "Up"; then
        log_success "Containers are running"
    else
        log_warn "Some containers may have issues"
    fi
    
    # Check API
    if command -v curl >/dev/null 2>&1; then
        if curl -s http://localhost:8001/health >/dev/null; then
            log_success "Search API is responding"
        else
            log_warn "Search API not responding yet"
        fi
        
        if curl -s http://localhost:5678 >/dev/null; then
            log_success "N8N is accessible"
        else
            log_warn "N8N not accessible yet"
        fi
        
        if curl -s http://localhost:8080 >/dev/null; then
            log_success "Open WebUI is accessible"
        else
            log_warn "Open WebUI not accessible yet"
        fi
    fi
}

# Load Ollama models
install_ollama_models() {
    if [ "$GPU_SUPPORT" = false ]; then
        log_info "Skipping Ollama models (CPU mode)"
        return
    fi
    
    log_info "Installing Ollama models..."
    
    local retries=0
    while [ $retries -lt 30 ]; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            break
        fi
        sleep 2
        ((retries++))
    done
    
    if [ $retries -eq 30 ]; then
        log_warn "Ollama not ready - skipping model installation"
        return
    fi
    
    models=("mistral" "llama3")
    for model in "${models[@]}"; do
        log_info "Installing model: $model"
        docker exec doc-intel-ollama ollama pull "$model" || log_warn "Failed to install $model"
    done
}

# Create test document
create_test_document() {
    log_info "Creating test document..."
    
    cat > data/test_installation.txt << EOF
Document Intelligence System Test

Installation completed: $(date)

Features tested:
- Automatic file detection
- OCR processing
- GDPR-compliant indexing
- Intelligent search

Test keywords: installation, test, system
Contact: test@example.com
Phone: +49 123 456789

This document will be automatically processed by the system.
EOF
    
    log_success "Test document created"
}

# Show summary
show_summary() {
    echo ""
    echo "================================================================"
    echo -e "${GREEN}✅ INSTALLATION COMPLETED SUCCESSFULLY!${NC}"
    echo "================================================================"
    echo ""
    echo -e "${BLUE}🌐 ACCESS POINTS:${NC}"
    echo "  • N8N Workflows:  http://localhost:5678 (admin/changeme)"
    echo "  • Open WebUI:     http://localhost:8080"
    echo "  • Search API:     http://localhost:8001/docs"
    echo "  • Health Check:   http://localhost:8001/health"
    echo ""
    echo -e "${BLUE}📋 NEXT STEPS:${NC}"
    echo "  1. Add documents to ./data folder"
    echo "  2. Open http://localhost:8080 for search interface"
    echo "  3. Configure N8N workflows at http://localhost:5678"
    echo "  4. Monitor logs: docker compose logs -f"
    echo ""
    echo -e "${BLUE}🔧 USEFUL COMMANDS:${NC}"
    echo "  • View logs:        docker compose logs -f"
    echo "  • Service status:   docker compose ps"
    echo "  • Stop services:    docker compose down"
    echo "  • Restart:          docker compose restart"
    echo ""
}

# Main installation
main() {
    echo ""
    log_info "Starting Document Intelligence System installation..."
    
    detect_system
    check_permissions
    check_project_directory
    
    read -p "Install system dependencies automatically? (y/n): " install_deps
    if [[ "$install_deps" =~ ^[Yy]$ ]]; then
        install_system_deps
    fi
    
    check_software
    check_gpu
    create_directories
    setup_environment
    install_python_deps
    start_services
    test_services
    
    if [ "$GPU_SUPPORT" = true ]; then
        install_ollama_models
    fi
    
    create_test_document
    show_summary
    
    log_success "Installation completed successfully!"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
