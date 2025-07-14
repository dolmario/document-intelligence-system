#!/usr/bin/env python3
"""
Schneller Fix für die Logger-Pfad-Probleme
Behebt nur die kritischen Fehler ohne Überkomplizierung
"""

import os
import re
from pathlib import Path

def fix_logger_calls():
    """Repariere Logger-Aufrufe in Python-Dateien"""
    print("🔧 Repariere Logger-Aufrufe...")
    
    # Suche nach Python-Dateien
    python_files = list(Path(".").rglob("*.py"))
    
    fixes = {
        "setup_logger('ocr_agent', 'logs/ocr.log')": "setup_logger('ocr_agent', 'ocr.log')",
        "setup_logger('search_api', 'logs/search_api.log')": "setup_logger('search_api', 'search_api.log')",
        "setup_logger('indexer_agent', 'logs/indexer.log')": "setup_logger('indexer_agent', 'indexer.log')",
        "setup_logger('watchdog_agent', 'logs/watchdog.log')": "setup_logger('watchdog_agent', 'watchdog.log')",
        "setup_logger('learning_agent', 'logs/learning.log')": "setup_logger('learning_agent', 'learning.log')"
    }
    
    fixed_files = 0
    
    for py_file in python_files:
        try:
            content = py_file.read_text(encoding='utf-8')
            original_content = content
            
            # Wende Fixes an
            for old, new in fixes.items():
                content = content.replace(old, new)
            
            # Speichere nur wenn geändert
            if content != original_content:
                py_file.write_text(content, encoding='utf-8')
                print(f"✅ Repariert: {py_file}")
                fixed_files += 1
                
        except Exception as e:
            print(f"⚠️ Fehler bei {py_file}: {e}")
    
    print(f"✅ {fixed_files} Dateien repariert")

def fix_env_file():
    """Füge fehlende Umgebungsvariablen hinzu"""
    print("🔧 Repariere .env Datei...")
    
    env_file = Path(".env")
    
    # Erstelle .env falls nicht vorhanden
    if not env_file.exists():
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ .env aus .env.example erstellt")
        else:
            env_file.write_text("""# Document Intelligence System
DATA_PATH=./data
INDEX_PATH=./indices
LOG_PATH=./logs
REDIS_URL=redis://redis:6379
CORS_ALLOW_ORIGIN=http://localhost:8080
USER_AGENT=DocumentIntelligenceSystem/1.0
CHROMA_TELEMETRY=false
""")
            print("✅ .env Datei erstellt")
            return
    
    # Lese bestehende .env
    content = env_file.read_text()
    
    # Füge fehlende Variablen hinzu
    missing_vars = {
        "CORS_ALLOW_ORIGIN": "http://localhost:8080,http://localhost:5678",
        "USER_AGENT": "DocumentIntelligenceSystem/1.0",
        "CHROMA_TELEMETRY": "false",
        "TORCH_VERSION": "2.1.0"
    }
    
    lines = content.split('\n')
    
    for var, value in missing_vars.items():
        # Prüfe ob Variable bereits existiert
        found = any(line.startswith(f"{var}=") for line in lines)
        
        if not found:
            lines.append(f"{var}={value}")
            print(f"✅ Hinzugefügt: {var}={value}")
    
    env_file.write_text('\n'.join(lines))

def fix_requirements():
    """Fixe requirements.txt für PyTorch"""
    print("🔧 Repariere requirements.txt...")
    
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt nicht gefunden")
        return
    
    content = req_file.read_text()
    
    # Ersetze PyTorch-Versionen
    content = re.sub(r'torch==[\d.]+', 'torch==2.1.0', content)
    content = re.sub(r'torchvision==[\d.]+', 'torchvision==0.16.0', content) 
    content = re.sub(r'torchaudio==[\d.]+', 'torchaudio==2.1.0', content)
    
    req_file.write_text(content)
    print("✅ PyTorch-Versionen korrigiert")

def restart_docker():
    """Starte Docker Services neu"""
    print("🔧 Starte Docker Services neu...")
    
    import subprocess
    
    try:
        # Services stoppen
        subprocess.run(['docker', 'compose', 'down'], check=False)
        print("✅ Services gestoppt")
        
        # Services starten
        subprocess.run(['docker', 'compose', 'up', '-d'], check=True)
        print("✅ Services gestartet")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker Fehler: {e}")
    except FileNotFoundError:
        print("❌ Docker nicht gefunden")

def main():
    """Hauptfunktion - nur die wichtigsten Fixes"""
    print("🚀 Schneller Fix für Document Intelligence System")
    print("=" * 50)
    
    # Prüfe ob wir im richtigen Verzeichnis sind
    if not Path("docker-compose.yml").exists():
        print("❌ Nicht im Projekt-Verzeichnis!")
        return
    
    try:
        fix_logger_calls()
        fix_env_file() 
        fix_requirements()
        
        restart = input("\nDocker Services neu starten? (j/n): ")
        if restart.lower() in ['j', 'y', 'ja', 'yes']:
            restart_docker()
        
        print("\n✅ Schnelle Fixes angewendet!")
        print("\nJetzt sollten die Installationsskripte funktionieren:")
        print("  - Doppelklick auf install.bat")
        print("  - Oder: .\\install.ps1 in PowerShell")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    main()
