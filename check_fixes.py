#!/usr/bin/env python3
"""
Health Check Script für Document Intelligence System
Überprüft ob alle Fixes korrekt angewendet wurden
"""

import subprocess
import requests
import json
import os
from pathlib import Path
import time

def check_docker_services():
    """Prüfe Docker Services"""
    print("\n🐳 DOCKER SERVICES CHECK")
    print("-" * 40)
    
    try:
        result = subprocess.run(['docker', 'compose', 'ps'], 
                              capture_output=True, text=True, check=True)
        print("✅ Docker Compose läuft")
        print(result.stdout)
        
        # Prüfe spezifische Services
        services = ['redis', 'search_api', 'ocr_agent', 'indexer', 'watchdog']
        for service in services:
            if service in result.stdout and 'Up' in result.stdout:
                print(f"✅ {service} läuft")
            else:
                print(f"❌ {service} läuft nicht")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker Compose Fehler: {e}")
        return False

def check_api_endpoints():
    """Prüfe API Endpoints"""
    print("\n🌐 API ENDPOINTS CHECK")
    print("-" * 40)
    
    endpoints = [
        ("Search API Health", "http://localhost:8001/health"),
        ("Search API Root", "http://localhost:8001/"),
        ("N8N", "http://localhost:5678"),
        ("Open WebUI", "http://localhost:8080"),
    ]
    
    results = {}
    
    for name, url in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: {response.status_code}")
                results[name] = True
            else:
                print(f"⚠️ {name}: {response.status_code}")
                results[name] = False
        except requests.exceptions.RequestException as e:
            print(f"❌ {name}: {str(e)}")
            results[name] = False
    
    return all(results.values())

def check_logs_for_errors():
    """Prüfe Docker Logs auf kritische Fehler"""
    print("\n📋 LOGS CHECK")
    print("-" * 40)
    
    try:
        # Hole letzte Logs
        result = subprocess.run(['docker', 'compose', 'logs', '--tail=50'], 
                              capture_output=True, text=True, check=True)
        
        logs = result.stdout.lower()
        
        # Prüfe auf kritische Fehler
        critical_errors = [
            'filenotfounderror',
            'no such file or directory',
            'connection refused',
            'failed to',
            'error:'
        ]
        
        error_count = 0
        for error in critical_errors:
            count = logs.count(error)
            if count > 0:
                print(f"⚠️ '{error}' gefunden: {count}x")
                error_count += count
        
        # Prüfe auf spezifische Fixes
        fixed_issues = [
            ('cors_allow_origin', 'CORS korrekt konfiguriert'),
            ('user_agent', 'User Agent gesetzt'),
            ('torch==2.1.0', 'PyTorch Version korrekt')
        ]
        
        for issue, description in fixed_issues:
            if issue in logs:
                print(f"✅ {description}")
            else:
                print(f"ℹ️ {description} nicht in Logs sichtbar")
        
        if error_count == 0:
            print("✅ Keine kritischen Fehler in Logs")
            return True
        else:
            print(f"⚠️ {error_count} potentielle Probleme gefunden")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Logs Check Fehler: {e}")
        return False

def check_environment_variables():
    """Prüfe .env Konfiguration"""
    print("\n⚙️ ENVIRONMENT VARIABLES CHECK")
    print("-" * 40)
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env Datei nicht gefunden")
        return False
    
    content = env_file.read_text()
    
    required_vars = {
        'CORS_ALLOW_ORIGIN': 'CORS Origins',
        'USER_AGENT': 'User Agent',
        'CHROMA_TELEMETRY': 'ChromaDB Telemetrie',
        'TORCH_VERSION': 'PyTorch Version',
        'LOG_PATH': 'Log Pfad'
    }
    
    all_found = True
    for var, description in required_vars.items():
        if f"{var}=" in content:
            print(f"✅ {description} konfiguriert")
        else:
            print(f"❌ {description} fehlt")
            all_found = False
    
    return all_found

def check_python_dependencies():
    """Prüfe Python Dependencies"""
    print("\n🐍 PYTHON DEPENDENCIES CHECK")
    print("-" * 40)
    
    # Prüfe ob venv existiert
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual Environment nicht gefunden")
        return False
    
    # Prüfe PyTorch Version
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Linux/Mac
        pip_cmd = "venv/bin/pip"
    
    try:
        result = subprocess.run([pip_cmd, 'show', 'torch'], 
                              capture_output=True, text=True, check=True)
        
        if "Version: 2.1.0" in result.stdout:
            print("✅ PyTorch 2.1.0 installiert")
        else:
            print("⚠️ PyTorch Version möglicherweise falsch")
            print(result.stdout)
        
        # Prüfe andere wichtige Pakete
        packages = ['spacy', 'fastapi', 'redis', 'sentence-transformers']
        for package in packages:
            try:
                subprocess.run([pip_cmd, 'show', package], 
                             capture_output=True, text=True, check=True)
                print(f"✅ {package} installiert")
            except subprocess.CalledProcessError:
                print(f"❌ {package} fehlt")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Dependency Check Fehler: {e}")
        return False

def test_search_functionality():
    """Teste Suchfunktionalität"""
    print("\n🔍 SEARCH FUNCTIONALITY TEST")
    print("-" * 40)
    
    try:
        # Teste einfache Suche
        search_data = {
            "query": "test",
            "limit": 5
        }
        
        response = requests.post(
            "http://localhost:8001/search",
            json=search_data,
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Suche funktional - {len(results)} Ergebnisse")
            return True
        else:
            print(f"⚠️ Suche Statuscode: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Such-Test Fehler: {str(e)}")
        return False

def generate_test_document():
    """Erstelle Test-Dokument"""
    print("\n📄 TEST DOCUMENT CREATION")
    print("-" * 40)
    
    test_doc_path = Path("data/health_check_test.txt")
    
    # Stelle sicher dass data Verzeichnis existiert
    test_doc_path.parent.mkdir(exist_ok=True)
    
    test_content = f"""Health Check Test Document
Created: {time.strftime('%Y-%m-%d %H:%M:%S')}

This is a test document to verify the Document Intelligence System is working correctly.

Features being tested:
- File detection by Watchdog
- OCR processing 
- Index generation
- Search functionality

Test keywords: health check, system test, verification
"""
    
    test_doc_path.write_text(test_content, encoding='utf-8')
    print(f"✅ Test-Dokument erstellt: {test_doc_path}")
    
    # Warte kurz auf Verarbeitung
    print("⏳ Warte 10 Sekunden auf Verarbeitung...")
    time.sleep(10)
    
    # Teste ob Dokument indiziert wurde
    try:
        response = requests.post(
            "http://localhost:8001/search",
            json={"query": "health check", "limit": 10},
            timeout=5
        )
        
        if response.status_code == 200:
            results = response.json()
            found = any("health" in r.get("title", "").lower() or 
                       "health" in r.get("snippet", "").lower() 
                       for r in results)
            
            if found:
                print("✅ Test-Dokument wurde erfolgreich indiziert und gefunden")
                return True
            else:
                print("⚠️ Test-Dokument nicht in Suchergebnissen gefunden")
                return False
        else:
            print("❌ Suche nach Test-Dokument fehlgeschlagen")
            return False
            
    except Exception as e:
        print(f"❌ Test-Dokument Verifikation Fehler: {e}")
        return False

def main():
    """Hauptfunktion - führe alle Health Checks aus"""
    print("🏥 Document Intelligence System - Health Check")
    print("=" * 60)
    
    checks = [
        ("Docker Services", check_docker_services),
        ("API Endpoints", check_api_endpoints),
        ("Environment Variables", check_environment_variables),
        ("Python Dependencies", check_python_dependencies),
        ("Log Analysis", check_logs_for_errors),
        ("Search Functionality", test_search_functionality),
        ("End-to-End Test", generate_test_document)
    ]
    
    results = {}
    
    for name, check_func in checks:
        print(f"\n🔍 Prüfe: {name}")
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Fehler bei {name}: {e}")
            results[name] = False
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("HEALTH CHECK ZUSAMMENFASSUNG")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<25} {status}")
    
    print(f"\nGesamt: {passed}/{total} Tests bestanden")
    
    if passed == total:
        print("\n🎉 Alle Health Checks erfolgreich!")
        print("Das System läuft einwandfrei.")
    elif passed >= total * 0.8:
        print("\n⚠️ System läuft größtenteils korrekt.")
        print("Einige kleinere Probleme könnten bestehen.")
    else:
        print("\n❌ Kritische Probleme erkannt.")
        print("Führe fix_docker_issues.py aus oder prüfe die Logs.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
