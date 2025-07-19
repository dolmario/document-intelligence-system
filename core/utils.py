import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import aiofiles
import asyncio


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Setup configured logger with proper path handling"""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_path = os.getenv('LOG_PATH', './logs')
        log_dir = Path(log_path)
        log_dir.mkdir(parents=True, exist_ok=True)

        full_log_path = log_dir / log_file
        
        try:
            file_handler = logging.FileHandler(full_log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create log file: {e}")

    return logger


def get_file_extension(filepath: str) -> str:
    """Extract file extension"""
    return Path(filepath).suffix.lower().lstrip('.')


def is_supported_file(filepath: str, supported_extensions: List[str]) -> bool:
    """Check if file is supported"""
    ext = get_file_extension(filepath)
    return ext in supported_extensions


async def read_file_async(filepath: str) -> bytes:
    """Read file asynchronously"""
    async with aiofiles.open(filepath, 'rb') as f:
        return await f.read()


async def write_file_async(filepath: str, content: bytes):
    """Write file asynchronously"""
    async with aiofiles.open(filepath, 'wb') as f:
        await f.write(content)


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(filepath: str, data: Dict[str, Any]):
    """Save JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to split at word boundary
        if end < len(text):
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.8:
                end = start + last_space
                chunk = text[start:end]

        chunks.append(chunk)
        start = end - overlap

    return chunks


def extract_text_statistics(text: str) -> Dict[str, int]:
    """Extract basic text statistics"""
    words = text.split()
    sentences = text.split('.')

    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
    }


def ensure_directory(path: Path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Find project root directory"""
    current = Path(__file__).parent
    while current.parent != current:
        if (current / '.git').exists() or (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    return Path.cwd()


class Config:
    """Central configuration management"""

    def __init__(self):
        self.load_from_env()

    def load_from_env(self):
        """Load configuration from environment variables"""
        self.data_path = Path(os.getenv('DATA_PATH', './data'))
        self.index_path = Path(os.getenv('INDEX_PATH', './indices'))
        self.log_path = Path(os.getenv('LOG_PATH', './logs'))
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.ocr_engine = os.getenv('OCR_ENGINE', 'tesseract')
        self.tesseract_lang = os.getenv('TESSERACT_LANG', 'deu+eng')
        self.enable_pii_removal = os.getenv('ENABLE_PII_REMOVAL', 'true').lower() == 'true'
        self.min_relevance_score = float(os.getenv('MIN_RELEVANCE_SCORE', '0.3'))
        self.cache_ttl = int(os.getenv('CACHE_TTL', '3600'))

        # Ensure directories exist
        ensure_directory(self.data_path)
        ensure_directory(self.index_path)
        ensure_directory(self.log_path)
        ensure_directory(self.index_path / 'markdown')
        ensure_directory(self.index_path / 'json')


# Singleton Config Instance
config = Config()
