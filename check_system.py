#!/usr/bin/env python3
"""System Check für Document Intelligence System"""

import sys
import subprocess
import platform
import os
from pathlib import Path

def check_python():
    """Check Python Version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 10:
        print("❌ Python 3.10+ benötigt!")
        return False
    return True

def check_docker():
    """Check Docker Installation"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        print(f"✓ Docker: {result.stdout.strip()}")
        
        # Check if Docker is running
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker läuft nicht!")
            return False
        return True
    except FileNotFoundError:
        print("❌ Docker nicht installiert!")
        return False

def check_gpu():
    """Check GPU Support"""
    try:
        # Check NVIDIA GPU
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU gefunden")
            
            # Check CUDA
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA verfügbar: {torch.cuda.get_device_name(0)}")
                return True
            else:
                print("⚠ CUDA nicht verfügbar")
    except:
        print("⚠ Keine GPU gefunden - CPU Mode")
    
    return False

def check_dependencies():
    """Check Python Dependencies"""
    required = [
        'redis', 'fastapi', 'spacy', 'pytesseract', 
        'transformers', 'sentence_transformers'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg} installiert")
        except ImportError:
            print(f"❌ {pkg} fehlt")
            missing.append(pkg)
    
    return len(missing) == 0

def check_directories():
    """Check Required Directories"""
    dirs = ['data', 'indices', 'logs', 'n8n/workflows']
    
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Erstellt: {dir_path}")
        else:
            print(f"✓ Vorhanden: {dir_path}")
    
    return True

def check_models():
    """Check Spacy Models"""
    try:
        import spacy
        nlp = spacy.load("de_core_news_sm")
        print("✓ Spacy Deutsch-Modell geladen")
        return True
    except:
        print("❌ Spacy Deutsch-Modell fehlt")
        print("  Installiere mit: python -m spacy download de_core_news_sm")
        return False

def main():
    """Run all checks"""
    print("=== Document Intelligence System Check ===\n")
    
    checks = [
        ("Python Version", check_python),
        ("Docker", check_docker),
        ("GPU Support", check_gpu),
        ("Verzeichnisse", check_directories),
        ("Dependencies", check_dependencies),
        ("Spacy Models", check_models),
    ]
    
    all_ok = True
    for name, check_func in checks:
        print(f"\n{name}:")
        if not check_func():
            all_ok = False
    
    print("\n" + "="*40)
    if all_ok:
        print("✅ System bereit!")
    else:
        print("❌ Einige Checks fehlgeschlagen - bitte beheben")
        sys.exit(1)

if __name__ == "__main__":
    main()
