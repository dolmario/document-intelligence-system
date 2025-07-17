#!/bin/bash

echo "🧪 Testing Document Intelligence System"
echo "======================================"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Test function
test_service() {
    local name=$1
    local url=$2
    local expected=$3
    
    echo -n "Testing $name... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    
    if [ "$response" = "$expected" ]; then
        echo -e "${GREEN}✅ OK (HTTP $response)${NC}"
        return 0
    else
        echo -e "${RED}❌ FAILED (HTTP $response, expected $expected)${NC}"
        return 1
    fi
}

# Run tests
echo -e "\n📡 Testing API endpoints:"
test_service "Search API" "http://localhost:8001/" "200"
test_service "N8N" "http://localhost:5678/" "200"
test_service "Open WebUI" "http://localhost:8080/" "200"
test_service "Qdrant" "http://localhost:6333/" "200"
test_service "Ollama" "http://localhost:11434/" "200"

# Test document upload
echo -e "\n📄 Testing document upload:"
response=$(curl -s -X POST http://localhost:8001/upload \
    -H "Content-Type: application/json" \
    -d '{"name":"test.txt","content":"This is a test document for the system."}')

if echo "$response" | grep -q "document_id"; then
    echo -e "${GREEN}✅ Upload successful${NC}"
    doc_id=$(echo "$response" | grep -o '"document_id":"[^"]*' | cut -d'"' -f4)
    echo "Document ID: $doc_id"
else
    echo -e "${RED}❌ Upload failed${NC}"
    echo "Response: $response"
fi

# Test search
echo -e "\n🔍 Testing search:"
sleep 5  # Wait for processing
search_response=$(curl -s -X POST http://localhost:8001/search \
    -H "Content-Type: application/json" \
    -d '{"query":"test document","limit":10}')

if echo "$search_response" | grep -q "\["; then
    echo -e "${GREEN}✅ Search working${NC}"
else
    echo -e "${RED}❌ Search failed${NC}"
    echo "Response: $search_response"
fi

# Check container status
echo -e "\n📊 Container status:"
docker compose ps

echo -e "\n✨ Test complete!"
