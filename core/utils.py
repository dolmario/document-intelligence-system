import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import aiofiles
import asyncio

# Logger Setup
def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Erstelle konfigurierten Logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler wenn angegeben
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# File Utils
def get_file_extension(filepath: str) -> str:
    """Extrahiere Dateierweiterung"""
    return Path(filepath).suffix.lower().lstrip('.')

def is_supported_file(filepath: str, supported_extensions: List[str]) -> bool:
    """Prüfe ob Datei unterstützt wird"""
    ext = get_file_extension(filepath)
    return ext in supported_extensions

async def read_file_async(filepath: str) -> bytes:
    """Lese Datei asynchron"""
    async with aiofiles.open(filepath, 'rb') as f:
        return await f.read()

async def write_file_async(filepath: str, content: bytes):
    """Schreibe Datei asynchron"""
    async with aiofiles.open(filepath, 'wb') as f:
        await f.write(content)

# JSON Utils
def load_json_file(filepath: str) -> Dict[str, Any]:
    """Lade JSON Datei"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(filepath: str, data: Dict[str, Any]):
    """Speichere JSON Datei"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Text Processing Utils
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Teile Text in überlappende Chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Versuche an Wortgrenze zu splitten
        if end < len(text):
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.8:  # Nur wenn nicht zu viel verloren geht
                end = start + last_space
                chunk = text[start:end]
        
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def extract_text_statistics(text: str) -> Dict[str, int]:
    """Extrahiere Basis-Statistiken aus Text"""
    words = text.split()
    sentences = text.split('.')
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
    }

# Path Utils
def ensure_directory(path: Path):
    """Stelle sicher dass Verzeichnis existiert"""
    path.mkdir(parents=True, exist_ok=True)

def get_project_root() -> Path:
    """Finde Projekt-Root-Verzeichnis"""
    current = Path(__file__).parent
    while current.parent != current:
        if (current / '.git').exists() or (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    return Path.cwd()

# Config Utils
class Config:
    """Zentrale Konfiguration"""
    
    def __init__(self):
        self.load_from_env()
    
    def load_from_env(self):
        """Lade Konfiguration aus Umgebungsvariablen"""
        self.data_path = Path(os.getenv('DATA_PATH', './data'))
        self.index_path = Path(os.getenv('INDEX_PATH', './indices'))
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.ocr_engine = os.getenv('OCR_ENGINE', 'tesseract')
        self.tesseract_lang = os.getenv('TESSERACT_LANG', 'deu+eng')
        self.enable_pii_removal = os.getenv('ENABLE_PII_REMOVAL', 'true').lower() == 'true'
        self.min_relevance_score = float(os.getenv('MIN_RELEVANCE_SCORE', '0.3'))
        self.cache_ttl = int(os.getenv('CACHE_TTL', '3600'))
        
        # Stelle sicher dass Verzeichnisse existieren
        ensure_directory(self.data_path)
        ensure_directory(self.index_path)
        ensure_directory(self.index_path / 'markdown')
        ensure_directory(self.index_path / 'json')

# Singleton Config Instance
config = Config()
