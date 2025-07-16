#!/bin/bash
set -e

echo "📦 Document Intelligence System V2 - Installation"
echo "=============================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found! Please install Docker first.${NC}"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found!${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running! Please start Docker.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"

# Create directories
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p data logs n8n/workflows
chmod -R 755 data logs n8n
touch data/.gitkeep logs/.gitkeep

# Setup environment
echo -e "\n${YELLOW}Setting up environment...${NC}"
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env from template${NC}"
        echo -e "${YELLOW}⚠️  Please edit .env to set secure passwords!${NC}"
    else
        echo -e "${RED}❌ .env.example not found!${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ .env already exists${NC}"
fi

# Stop any existing containers
echo -e "\n${YELLOW}Cleaning up old containers...${NC}"
docker compose down 2>/dev/null || true

# Build containers
echo -e "\n${YELLOW}Building containers (this may take 5-10 minutes)...${NC}"
docker compose build --no-cache

# Start services in order
echo -e "\n${YELLOW}Starting services...${NC}"

# Start PostgreSQL first
echo "Starting PostgreSQL..."
docker compose up -d postgres
echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Wait for PostgreSQL to be healthy
RETRIES=30
until docker compose exec -T postgres pg_isready -U docintell -d document_intelligence &>/dev/null; do
    RETRIES=$((RETRIES-1))
    if [ $RETRIES -le 0 ]; then
        echo -e "${RED}❌ PostgreSQL failed to start!${NC}"
        docker compose logs postgres
        exit 1
    fi
    echo -n "."
    sleep 2
done
echo -e "\n${GREEN}✅ PostgreSQL ready${NC}"

# Start other services
echo "Starting remaining services..."
docker compose up -d

# Wait for all services
echo -e "\n${YELLOW}Waiting for all services to be ready...${NC}"
sleep 20

# Check service status
echo -e "\n${YELLOW}Checking service status...${NC}"
SERVICES=("postgres" "qdrant" "ocr_agent" "search_api" "n8n" "ollama" "open-webui")
ALL_OK=true

for service in "${SERVICES[@]}"; do
    if docker compose ps | grep -q "doc-intel-$service.*Up\|doc-intel-$service.*running"; then
        echo -e "${GREEN}✅ $service is running${NC}"
    else
        echo -e "${RED}❌ $service is not running${NC}"
        ALL_OK=false
    fi
done

# Load default model
echo -e "\n${YELLOW}Loading default LLM model (mistral)...${NC}"
docker exec doc-intel-ollama ollama pull mistral 2>/dev/null || echo -e "${YELLOW}⚠️  Model loading failed - you can do this later${NC}"

# Final status
if [ "$ALL_OK" = true ]; then
    echo -e "\n${GREEN}✅ Installation complete!${NC}"
else
    echo -e "\n${YELLOW}⚠️  Some services failed to start. Check logs with: docker compose logs${NC}"
fi

echo -e "\n${GREEN}🌐 Access Points:${NC}"
echo "   • N8N Workflows:  http://localhost:5678 (admin/changeme)"
echo "   • Search API:     http://localhost:8001"
echo "   • Open WebUI:     http://localhost:8080"
echo "   • Qdrant UI:      http://localhost:6333/dashboard"

echo -e "\n${GREEN}📝 Next Steps:${NC}"
echo "   1. Change default passwords in .env"
echo "   2. Import workflows: n8n/workflows/document_processing_v2.json"
echo "   3. Test with: curl -X POST http://localhost:8001/upload -H 'Content-Type: application/json' -d '{\"name\":\"test.txt\",\"content\":\"Hello World\"}'"

echo -e "\n${GREEN}🤖 To add more models:${NC}"
echo "   docker exec doc-intel-ollama ollama pull llama2"
echo "   docker exec doc-intel-ollama ollama pull codellama"

echo -e "\n${GREEN}📚 View logs:${NC}"
echo "   docker compose logs -f"

echo -e "\n${GREEN}🛑 To stop:${NC}"
echo "   docker compose down"
