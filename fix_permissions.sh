#!/bin/bash
# Fix permissions for Linux/Mac if needed

echo "Fixing permissions..."

# Create directories if missing
mkdir -p data indices logs n8n/workflows
mkdir -p indices/json indices/markdown

# Set permissions
chmod -R 755 data indices logs n8n

# Create .gitkeep files
touch data/.gitkeep indices/.gitkeep logs/.gitkeep

echo "Permissions fixed!"
