#!/bin/bash

echo "🔧 Quick Fix for Document Intelligence System"
echo "==========================================="

# Fix 1: Rebuild Search API with correct code
echo "Rebuilding Search API..."
docker compose stop search_api
docker compose build --no-cache search_api
docker compose up -d search_api

# Fix 2: Wait for services
echo "Waiting for services..."
sleep 10

# Fix 3: Test upload with simple document
echo -e "\nTesting system..."
chmod +x simple_test.sh
./simple_test.sh

echo -e "\n✅ Quick fix applied!"
echo ""
echo "⚠️  Don't forget to:"
echo "1. Activate N8N workflow (see N8N_SETUP.md)"
echo "2. Clean up old files (see CLEANUP.md)"
echo "3. Change passwords in .env for production"
