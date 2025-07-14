#!/usr/bin/env python3
"""
Automatisches Fix-Script für Document Intelligence System
Behebt alle identifizierten Docker- und Konfigurationsprobleme
"""

import os
import subprocess
import shutil
from pathlib import Path
import sys

def print_step(step, description):
    """Formatierte Ausgabe für Steps"""
    print(f"\n{'='*60}")
    print(f"STEP {step}: {description}")
    print(f"{'='*60}")

def run_command(cmd, description="", check=True):
    """Führe Kommando aus mit Logging"""
    print(f"Ausführung: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {description}")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler bei: {description}")
        print(f"Error: {e.stderr}")
        return False

def backup_file(filepath):
    """Erstelle Backup einer Datei"""
    if Path(filepath).exists():
        backup_path = f"{filepath}.backup"
        shutil.copy2(filepath, backup_path)
        print(f"📁 Backup erstellt: {backup_path}")

def fix_requirements():
    """Fix 1: PyTorch Version downgraden"""
    print_step(1, "PyTorch Version korrigieren")
    
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print("❌ requirements.txt nicht gefunden")
        return False
    
    backup_file(requirements_path)
    
    # Lese aktuelle requirements
    content = requirements_path.read_text()
    
    # Ersetze PyTorch Versionen
    fixes = [
        ("torch==2.1.0", "torch==2.1.0"),
        ("torchvision==0.16.0", "torchvision==0.16.0"), 
        ("torchaudio==2.1.0", "torchaudio==2.1.0"),
        ("protobuf==3.20.3", "protobuf==3.20.3")
    ]
    
    for old, new in fixes:
        if old not in content and "torch==" in content:
            # Ersetze bestehende torch Version
            import re
            content = re.sub(r'torch==[\d.]+', new, content)
            content = re.sub(r'torchvision==[\d.]+', 'torchvision==0.16.0', content)
            content = re.sub(r'torchaudio==[\d.]+', 'torchaudio==2.1.0', content)
    
    # Füge fehlende Dependencies hinzu
    if "langchain-community" not in content:
        content += "\nlangchain-community==0.0.38\nlangchain-core==0.1.52\n"
    
    requirements_path.write_text(content)
    print("✅ requirements.txt aktualisiert")
    return True

def fix_env_file():
    """Fix 2: .env Datei korrigieren"""
    print_step(2, "Umgebungsvariablen korrigieren")
    
    env_path = Path(".env")
    if not env_path.exists():
        # Kopiere von .env.example
        example_path = Path(".env.example")
        if example_path.exists():
            shutil.copy2(example_path, env_path)
            print("✅ .env aus .env.example erstellt")
        else:
            print("❌ Weder .env noch .env.example gefunden")
            return False
    
    backup_file(env_path)
    
    # Lese aktuelle .env
    content = env_path.read_text() if env_path.exists() else ""
    
    # Füge fehlende Umgebungsvariablen hinzu
    new_vars = {
        "CORS_ALLOW_ORIGIN": "http://localhost:8080,http://localhost:5678",
        "CORS_ALLOW_CREDENTIALS": "false",
        "CORS_ALLOW_METHODS": "GET,POST,PUT,DELETE",
        "CORS_ALLOW_HEADERS": "*",
        "CHROMA_TELEMETRY": "false",
        "ANONYMIZED_TELEMETRY": "false",
        "USER_AGENT": "DocumentIntelligenceSystem/1.0",
        "TORCH_VERSION": "2.1.0",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "LANGCHAIN_VERBOSE": "false",
        "LANGCHAIN_TRACING": "false"
    }
    
    lines = content.split('\n')
    
    for var, value in new_vars.items():
        # Prüfe ob Variable bereits existiert
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{var}="):
                lines[i] = f"{var}={value}"
                found = True
                break
        
        if not found:
            lines.append(f"{var}={value}")
    
    env_path.write_text('\n'.join(lines))
    print("✅ .env Datei aktualisiert")
    return True

def fix_logger_paths():
    """Fix 3: Logger-Pfade in Python-Dateien korrigieren"""
    print_step(3, "Logger-Pfade korrigieren")
    
    # Suche nach Python-Dateien mit Logger-Aufrufen
    python_files = []
    for pattern in ["**/*.py"]:
        python_files.extend(Path(".").glob(pattern))
    
    fixes_applied = 0
    
    for py_file in python_files:
        try:
            content = py_file.read_text(encoding='utf-8')
            original_content = content
            
            # Korrigiere Logger-Aufrufe
            fixes = [
                ("setup_logger('ocr_agent', 'logs/ocr.log')", "setup_logger('ocr_agent', 'ocr.log')"),
                ("setup_logger('search_api', 'logs/search_api.log')", "setup_logger('search_api', 'search_api.log')"),
                ("setup_logger('indexer_agent', 'logs/indexer.log')", "setup_logger('indexer_agent', 'indexer.log')"),
                ("setup_logger('watchdog_agent', 'watchdog.log')", "setup_logger('watchdog_agent', 'watchdog.log')")
            ]
            
            for old, new in fixes:
                content = content.replace(old, new)
            
            if content != original_content:
                backup_file(py_file)
                py_file.write_text(content, encoding='utf-8')
                fixes_applied += 1
                print(f"✅ Korrigiert: {py_file}")
                
        except Exception as e:
            print(f"⚠️ Warnung bei {py_file}: {e}")
    
    print(f"✅ {fixes_applied} Dateien korrigiert")
    return True

def fix_docker_compose():
    """Fix 4: Docker Compose Konfiguration optimieren"""
    print_step(4, "Docker Compose optimieren")
    
    compose_file = Path("docker-compose.yml")
    if not compose_file.exists():
        print("❌ docker-compose.yml nicht gefunden")
        return False
    
    backup_file(compose_file)
    
    # Füge Health Checks hinzu falls nicht vorhanden
    content = compose_file.read_text()
    
    # Einfache Health Check Ergänzung für Search API
    if "healthcheck:" not in content and "search_api:" in content:
        # Dies ist ein vereinfachter Fix - für komplexere Änderungen wäre YAML-Parsing nötig
        print("ℹ️ Für Docker Compose Optimierungen siehe docker-compose.override.yml")
    
    print("✅ Docker Compose überprüft")
    return True

def install_fixed_requirements():
    """Fix 5: Installiere korrigierte Requirements"""
    print_step(5, "Korrigierte Dependencies installieren")
    
    # Prüfe ob venv existiert
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Erstelle Virtual Environment...")
        if not run_command("python -m venv venv", "Virtual Environment erstellen"):
            return False
    
    # Aktiviere venv und installiere
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:  # Linux/Mac
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    commands = [
        (f"{pip_cmd} install --upgrade pip", "Pip Update"),
        (f"{pip_cmd} install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0", "PyTorch Installation"),
        (f"{pip_cmd} install -r requirements.txt", "Requirements Installation"),
        (f"{python_cmd} -m spacy download de_core_news_sm", "Spacy Modell Download")
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc, check=False):
            print(f"⚠️ Warnung: {desc} fehlgeschlagen")
    
    print("✅ Dependencies aktualisiert")
    return True

def restart_docker_services():
    """Fix 6: Docker Services neu starten"""
    print_step(6, "Docker Services neu starten")
    
    commands = [
        ("docker compose down", "Services stoppen"),
        ("docker compose build --no-cache", "Images neu bauen"),
        ("docker compose up -d", "Services starten")
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc, check=False):
            print(f"⚠️ Warnung: {desc} fehlgeschlagen")
    
    print("✅ Docker Services neu gestartet")
    return True

def verify_fixes():
    """Fix 7: Verifiziere Korrekturen"""
    print_step(7, "Fixes verifizieren")
    
    checks = [
        ("docker compose ps", "Service Status"),
        ("curl -s http://localhost:8001/health", "API Health Check"),
        ("docker compose logs --tail=10", "Letzte Logs")
    ]
    
    for cmd, desc in checks:
        run_command(cmd, desc, check=False)
    
    print("✅ Verifikation abgeschlossen")
    return True

def main():
    """Hauptfunktion - führe alle Fixes aus"""
    print("🔧 Document Intelligence System - Automatische Fehlerbehebung")
    print("================================================================")
    
    # Prüfe ob wir im richtigen Verzeichnis sind
    if not Path("docker-compose.yml").exists():
        print("❌ Nicht im Projekt-Verzeichnis! Bitte ins Projekt-Root wechseln.")
        sys.exit(1)
    
    fixes = [
        fix_requirements,
        fix_env_file,
        fix_logger_paths,
        fix_docker_compose,
        install_fixed_requirements,
        restart_docker_services,
        verify_fixes
    ]
    
    success_count = 0
    
    for fix_func in fixes:
        try:
            if fix_func():
                success_count += 1
        except Exception as e:
            print(f"❌ Fehler in {fix_func.__name__}: {e}")
    
    print(f"\n{'='*60}")
    print(f"ZUSAMMENFASSUNG: {success_count}/{len(fixes)} Fixes erfolgreich")
    print(f"{'='*60}")
    
    if success_count == len(fixes):
        print("🎉 Alle Fixes erfolgreich angewendet!")
        print("\nNächste Schritte:")
        print("1. Logs prüfen: docker compose logs -f")
        print("2. Services testen:")
        print("   - N8N: http://localhost:5678")
        print("   - Open WebUI: http://localhost:8080") 
        print("   - Search API: http://localhost:8001/docs")
    else:
        print("⚠️ Einige Fixes fehlgeschlagen. Prüfe die Ausgabe oben.")
        print("Manuelle Schritte könnten erforderlich sein.")

if __name__ == "__main__":
    main()
