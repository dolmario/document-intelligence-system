document-intelligence-system/
├── README.md
├── .gitignore
├── .env.example
├── docker-compose.yml
├── requirements.txt
│
├── agents/
│   ├── __init__.py
│   ├── watchdog/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── watchdog_agent.py
│   ├── ocr/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── ocr_agent.py
│   └── indexer/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── index_generator.py
│
├── core/
│   ├── __init__.py
│   ├── models.py          # DocumentIndex Datenmodelle
│   ├── privacy.py         # DSGVO-Funktionen
│   └── utils.py           # Gemeinsame Utilities
│
├── services/
│   ├── search/
│   │   ├── Dockerfile
│   │   ├── api.py
│   │   └── engine.py
│   └── learning/
│       ├── Dockerfile
│       └── learning_agent.py
│
├── n8n/
│   └── workflows/
│       └── document_processing.json
│
├── data/
│   └── .gitkeep
│
├── indices/
│   └── .gitkeep
│
└── tests/
    ├── __init__.py
    ├── test_ocr.py
    ├── test_indexer.py
    └── test_search.py
