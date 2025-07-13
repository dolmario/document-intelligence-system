#!/usr/bin/env python3
"""Startup Script für Document Intelligence System"""

import subprocess
import time
import requests
import sys
from pathlib import Path

def start_services():
    """Start Docker Services"""
    print("Starting Docker services...")
    subprocess.run(['docker', 'compose', 'up', '-d'])
    
    # Wait for services
    print("Waiting for services to start...")
    time.sleep(20)

def check_service(name, url):
    """Check if service is running"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✓ {name} running at {url}")
            return True
    except:
        pass
    
    print(f"❌ {name} not reachable at {url}")
    return False

def load_ollama_models():
    """Load Ollama models"""
    print("\nLoading Ollama models...")
    models = ['mistral', 'llama3', 'phi']
    
    for model in models:
        print(f"Loading {model}...")
        subprocess.run([
            'docker', 'exec', 'document-intelligence-system-ollama-1',
            'ollama', 'pull', model
        ])

def create_test_document():
    """Create test document"""
    test_doc = """Document Intelligence System Test

Dies ist ein Test-Dokument für das System.

Features:
- Automatische OCR
- DSGVO-konforme Verarbeitung
- Intelligente Suche
- KI-Unterstützung

Kontakt: test@example.com
Tel: +49 123 456789
"""
    
    Path('data/test_document.txt').write_text(test_doc, encoding='utf-8')
    print("✓ Test document created")

def main():
    """Main startup routine"""
    print("=== Starting Document Intelligence System ===\n")
    
    # Check system first
    check_result = subprocess.run([sys.executable, 'check_system.py'])
    if check_result.returncode != 0:
        print("System check failed! Please fix issues first.")
        sys.exit(1)
    
    # Start services
    start_services()
    
    # Check services
    services = [
        ("Redis", "http://localhost:6379"),
        ("N8N", "http://localhost:5678"),
        ("Search API", "http://localhost:8001"),
        ("Open WebUI", "http://localhost:8080"),
    ]
    
    print("\nChecking services...")
    all_ok = True
    for name, url in services:
        if not check_service(name, url):
            all_ok = False
    
    if not all_ok:
        print("\nSome services failed to start. Check logs with:")
        print("  docker compose logs")
        sys.exit(1)
    
    # Load models
    try:
        load_ollama_models()
    except:
        print("⚠ Could not load Ollama models")
    
    # Create test document
    create_test_document()
    
    print("\n" + "="*50)
    print("✅ System started successfully!")
    print("\nAccess points:")
    print("  - N8N:        http://localhost:5678 (admin/changeme)")
    print("  - Open WebUI: http://localhost:8080")
    print("  - Search API: http://localhost:8001/docs")
    print("\nNext steps:")
    print("  1. Add documents to ./data")
    print("  2. Open WebUI for search")
    print("  3. Configure N8N workflows")

if __name__ == "__main__":
    main()
