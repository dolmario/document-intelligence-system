#!/bin/bash

echo "=== Document Intelligence System Installation ==="
echo "================================================"

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Funktion für Fehlerbehandlung
error_exit() {
    echo -e "${RED}FEHLER: $1${NC}" >&2
    exit 1
}

# Funktion für Success-Meldungen
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Funktion für Warnings
warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# System-Check
echo "Checking System Requirements..."

# Python Check
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    success "Python gefunden: $PYTHON_VERSION"
else
    error_exit "Python 3 nicht gefunden. Bitte installieren."
fi

# Docker Check
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    success "Docker gefunden: $DOCKER_VERSION"
else
    error_exit "Docker nicht gefunden. Bitte installieren: https://docs.docker.com/get-docker/"
fi

# Docker Compose Check
if command -v docker compose &> /dev/null; then
    success "Docker Compose gefunden"
else
    error_exit "Docker Compose nicht gefunden."
fi

# NVIDIA GPU Check (optional)
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    success "NVIDIA GPU gefunden"
    # NVIDIA Container Toolkit Check
    if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        success "NVIDIA Container Toolkit installiert"
        GPU_SUPPORT=true
    else
        warning "NVIDIA Container Toolkit nicht installiert"
        echo "Für GPU-Support installieren mit:"
        echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        GPU_SUPPORT=false
    fi
else
    warning "Keine NVIDIA GPU gefunden - Ollama läuft auf CPU"
    GPU_SUPPORT=false
fi

# Erstelle Verzeichnisstruktur
echo ""
echo "Creating directory structure..."
mkdir -p data indices logs n8n/workflows
mkdir -p indices/{json,markdown}
touch data/.gitkeep indices/.gitkeep logs/.gitkeep

success "Verzeichnisse erstellt"

# Environment Setup
echo ""
echo "Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    success ".env Datei erstellt"
    warning "Bitte .env Datei anpassen!"
else
    warning ".env existiert bereits"
fi

# Python Dependencies
echo ""
echo "Installing Python dependencies..."
python3 -m venv venv
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

pip install --upgrade pip
pip install -r requirements.txt

# Spacy Model
echo ""
echo "Downloading Spacy language model..."
python -m spacy download de_core_news_sm

success "Python Dependencies installiert"

# Docker Images pullen
echo ""
echo "Pulling Docker images..."
docker compose pull

# Ollama Models vorbereiten
echo ""
echo "Preparing Ollama models..."
cat > setup_ollama.sh << 'EOF'
#!/bin/bash
# Warte bis Ollama bereit ist
echo "Warte auf Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done

echo "Ollama bereit. Lade Modelle..."

# Lade empfohlene Modelle
models=("mistral" "llama3" "phi")

for model in "${models[@]}"; do
    echo "Lade $model..."
    docker exec -it document-intelligence-system-ollama-1 ollama pull $model
done

echo "Modelle geladen!"
EOF

chmod +x setup_ollama.sh

# Starte Services
echo ""
echo "Starting services..."
if [ "$GPU_SUPPORT" = true ]; then
    docker compose up -d
else
    # Starte ohne GPU
    docker compose up -d --scale ollama=0
    warning "Ollama ohne GPU-Support gestartet"
fi

# Warte auf Services
echo ""
echo "Waiting for services to start..."
sleep 10

# Health Checks
echo ""
echo "Running health checks..."

# Redis
if curl -s http://localhost:6379 > /dev/null 2>&1; then
    success "Redis läuft"
else
    warning "Redis nicht erreichbar"
fi

# N8N
if curl -s http://localhost:5678 > /dev/null 2>&1; then
    success "N8N läuft auf http://localhost:5678"
else
    warning "N8N nicht erreichbar"
fi

# Search API
sleep 5
if curl -s http://localhost:8001 > /dev/null 2>&1; then
    success "Search API läuft auf http://localhost:8001"
else
    warning "Search API nicht erreichbar"
fi

# Open WebUI
if curl -s http://localhost:8080 > /dev/null 2>&1; then
    success "Open WebUI läuft auf http://localhost:8080"
else
    warning "Open WebUI nicht erreichbar"
fi

# Ollama Models Setup
if [ "$GPU_SUPPORT" = true ]; then
    echo ""
    echo "Setting up Ollama models..."
    ./setup_ollama.sh &
fi

# Final Output
echo ""
echo "=========================================="
echo -e "${GREEN}Installation abgeschlossen!${NC}"
echo "=========================================="
echo ""
echo "Services:"
echo "  - N8N:         http://localhost:5678 (admin/changeme)"
echo "  - Open WebUI:  http://localhost:8080"
echo "  - Search API:  http://localhost:8001/docs"
echo "  - Redis:       localhost:6379"
echo ""
echo "Nächste Schritte:"
echo "  1. Passe die .env Datei an"
echo "  2. Lege Dokumente in ./data ab"
echo "  3. Öffne Open WebUI für die Suche"
echo ""
echo "Logs anzeigen: docker compose logs -f"
echo "Stoppen:       docker compose down"
echo ""

# Test-Dokument erstellen
echo "Möchtest du ein Test-Dokument erstellen? (j/n)"
read -r response
if [[ "$response" =~ ^[Jj]$ ]]; then
    cat > data/test_document.txt << 'EOF'
Dies ist ein Test-Dokument für das Document Intelligence System.

Es enthält verschiedene Abschnitte und Informationen, die indiziert werden sollen.

Abschnitt 1: Einführung
Das System kann Dokumente automatisch verarbeiten, OCR durchführen und intelligent durchsuchen.

Abschnitt 2: Features
- Automatische Texterkennung
- DSGVO-konforme Verarbeitung
- Intelligente Verknüpfungen
- KI-gestützte Suche

Siehe auch: Benutzerhandbuch
Basiert auf: Document Intelligence Konzept
EOF
    success "Test-Dokument erstellt in data/test_document.txt"
fi

echo ""
echo "Installation komplett! 🎉"
