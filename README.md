Vollständige & saubere README.md (zum Kopieren) 
markdown
 
 
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
⌄
⌄
⌄
⌄
⌄
⌄
⌄
⌄
# 📁 Document Intelligence System (WIP 🚧)

Ein modulares, DSGVO-konformes Dokumenten- und Wissensmanagement-System mit intelligenter Volltextsuche, semantischer Verknüpfung, automatischer Indexierung und OCR-Unterstützung.

> ⚠️ **Dieses Projekt befindet sich noch im Aufbau.** Viele Funktionen sind experimentell oder unvollständig. Feedback und Beteiligung sind willkommen!

---

[![Tests](https://github.com/ [USERNAME]/document-intelligence-system/actions/workflows/test.yml/badge.svg)](https://github.com/ [USERNAME]/document-intelligence-system/actions/workflows/test.yml)
[![Docker](https://img.shields.io/docker/pulls/ [USERNAME]/doc-intelligence)](https://hub.docker.com/r/ [USERNAME]/doc-intelligence)
[![License](https://img.shields.io/badge/License-MIT-blue.svg )](LICENSE)

Ein modulares, DSGVO-konformes Dokumenten- und Wissensmanagement-System mit intelligenter Volltextsuche, semantischer Verknüpfung und KI-Unterstützung.

## 🚀 Features

- 🔍 **Intelligente Suche**: Volltext- und semantische Suche mit KI  
- 🤖 **Lokale OCR**: Tesseract-basierte Texterkennung  
- 🧠 **Lernfähig**: Verbessert sich durch Nutzungsmuster  
- 🔒 **DSGVO-konform**: Vollständig lokale Verarbeitung  
- 🔗 **Automatische Verknüpfungen**: Erkennt Zusammenhänge  
- 🎯 **N8N Integration**: Workflow-Automatisierung  
- 🤖 **Ollama Integration**: Lokale KI-Modelle  
- 📊 **Open WebUI**: Moderne Benutzeroberfläche  

## 📋 Voraussetzungen

| Komponente     | Anforderung                              |
|----------------|-------------------------------------------|
| **OS**         | Windows 11, Linux, macOS                  |
| **Python**     | 3.10+                                     |
| **Docker**     | Docker Desktop (Windows/Mac) oder Engine  |
| **RAM**        | Mindestens 8GB (16GB empfohlen)           |
| **GPU**        | NVIDIA GPU (optional)                     |
| **Speicher**   | 20GB+ freier Speicherplatz                |

## 🛠️ Installation

### Windows (PowerShell als Administrator)

```powershell
git clone https://github.com/ [USERNAME]/document-intelligence-system.git
cd document-intelligence-system
.\install.ps1
 
 
Linux/macOS 
bash
 
 
1
2
3
4
git clone https://github.com/ [USERNAME]/document-intelligence-system.git
cd document-intelligence-system
chmod +x install.sh
./install.sh
 
 
Manuelle Installation 
bash
 
 
1
2
3
4
5
6
7
8
9
10
11
12
13
14
# 1. Repository klonen
git clone https://github.com/ [USERNAME]/document-intelligence-system.git
cd document-intelligence-system

# 2. Environment konfigurieren
cp .env.example .env
# Bearbeite .env nach deinen Bedürfnissen

# 3. Services starten
docker compose up -d

# 4. Ollama Modelle laden (optional)
docker exec -it document-intelligence-system-ollama-1 ollama pull mistral
docker exec -it document-intelligence-system-ollama-1 ollama pull llama3
 
 
🎮 Verwendung 
Web-Interfaces 

    N8N : http://localhost:5678  (admin/changeme)
    Open WebUI : http://localhost:8080 
    Search API : http://localhost:8001/docs 
     

Dokumente hinzufügen 

    Lege Dokumente in den ./data Ordner
    Der Watchdog erkennt neue Dateien automatisch
    OCR und Indexierung erfolgen automatisch
    Dokumente sind sofort durchsuchbar
     

API Beispiele (Python) 
python
 
 
1
2
3
4
5
6
7
8
9
10
11
12
13
14
⌄
import requests

# Suche
response = requests.post('http://localhost:8001/search', json={
    'query': 'Machine Learning Python',
    'limit': 10
})
results = response.json()

# Dokument abrufen
doc = requests.get(f'http://localhost:8001/document/{doc_id}')

# Ähnliche Dokumente
similar = requests.get(f'http://localhost:8001/suggest/{doc_id}')
 
 
🧪 Testing 
bash
 
 
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
# Virtuelle Umgebung aktivieren
source venv/bin/activate  # Linux/macOS
# oder
.\venv\Scripts\Activate.ps1  # Windows

# Alle Tests
pytest

# Nur Unit Tests
pytest -m "not integration"

# Mit Coverage
pytest --cov --cov-report=html

# Spezifische Tests
pytest tests/test_ocr.py -v
 
 
🏗️ Architektur 
 
 
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
┌─────────────────────────────────────────────────────────┐
│                    Open WebUI (UI)                      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                   Search API (FastAPI)                   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                    Redis (Queue)                         │
└────┬──────────────┬──────────────┬──────────────┬──────┘
     │              │              │              │
┌────┴────┐    ┌───┴───┐    ┌────┴────┐    ┌────┴────┐
│Watchdog │    │  OCR  │    │ Indexer │    │Learning │
│ Agent   │    │ Agent │    │  Agent  │    │ Agent   │
└─────────┘    └───────┘    └─────────┘    └─────────┘
 
 
⚙️ Konfiguration 
env
 
 
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
# OCR
TESSERACT_LANG=deu+eng
OCR_ENGINE=tesseract

# Privacy
ENABLE_PII_REMOVAL=true
ANONYMIZE_METADATA=true

# Storage
DATA_PATH=./data
INDEX_PATH=./indices

# Redis
REDIS_URL=redis://redis:6379

# API
MIN_RELEVANCE_SCORE=0.3
CACHE_TTL=3600
 
 
📄 Unterstützte Dateiformate 

    Dokumente : PDF, DOCX, DOC, TXT, MD  
    Bilder : PNG, JPG, JPEG, TIFF, TIF  
    Weitere : Erweiterbar über Plugins
     

🚀 Erweiterte Features 
Manuelle Verknüpfungen 
bash
 
 
1
2
3
4
5
6
7
curl -X POST http://localhost:8001/link \
  -H "Content-Type: application/json" \
  -d '{
    "doc1_id": "abc123...",
    "doc2_id": "def456...",
    "bidirectional": true
  }'
 
 
N8N Workflows 

    Öffne N8N: http://localhost:5678   
    Importiere Workflows aus n8n/workflows/  
    Passe Webhook-URLs an  
    Aktiviere Workflows
     

Ollama Modelle 
bash
 
 
1
2
3
4
5
6
7
8
9
10
11
# Liste verfügbare Modelle
docker exec -it document-intelligence-system-ollama-1 ollama list

# Neues Modell laden
docker exec -it document-intelligence-system-ollama-1 ollama pull codellama

# Modell nutzen
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "Analysiere dieses Dokument..."
}'
 
 
📊 Monitoring 
Logs anzeigen 
bash
 
 
1
2
3
4
5
6
7
8
# Alle Services
docker compose logs -f

# Spezifischer Service
docker compose logs -f watchdog

# Letzte 100 Zeilen
docker compose logs --tail=100
 
 
System-Status 
bash
 
 
1
2
3
4
5
6
7
8
9
10
# Service Status
docker compose ps

# Ressourcen-Nutzung
docker stats

# Redis Queue Status
docker exec -it document-intelligence-system-redis-1 redis-cli
> LLEN processing_queue
> LLEN indexing_queue
 
 
🐛 Troubleshooting 
OCR erkennt keinen Text
	
Prüfe Bildqualität & Spracheinstellung, Logs prüfen
Keine GPU-Unterstützung
	
NVIDIA Container Toolkit installieren, Docker neu starten
Services starten nicht
	
docker compose down -v && docker compose up -d
 
 
🤝 Contributing 

    Fork das Repository
    Erstelle einen Feature Branch: git checkout -b feature/AmazingFeature
    Committe deine Änderungen: git commit -m 'Add some AmazingFeature'
    Push zum Branch: git push origin feature/AmazingFeature
    Öffne einen Pull Request
     

Development Setup 
bash
 
 
1
2
3
4
5
6
7
8
9
10
11
12
# Development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Code formatting
black .
isort .

# Type checking
mypy .
 
 
📝 Lizenz 

Dieses Projekt ist unter der MIT-Lizenz lizenziert – siehe LICENSE  für Details. 
🙏 Danksagungen 

    Tesseract OCR  
    Ollama  
    Open WebUI  
    N8N  
    spaCy
     

📞 Support 

    Issues: [GitHub Issues](https://github.com/  [USERNAME]/document-intelligence-system/issues)
    Discussions: [GitHub Discussions](https://github.com/  [USERNAME]/document-intelligence-system/discussions)
    Wiki: [Project Wiki](https://github.com/  [USERNAME]/document-intelligence-system/wiki)
     
     
