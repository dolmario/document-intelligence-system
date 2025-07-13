from pydantic import BaseSettings, Field
from typing import Optional
import os

class APIConfig(BaseSettings):
    """API Konfiguration mit Validierung"""
    
    # Allgemeine Einstellungen
    app_name: str = "Document Intelligence System"
    debug: bool = Field(default=False, env="DEBUG")
    
    # API Keys (für externe Services falls benötigt)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Ollama Einstellungen
    ollama_base_url: str = Field(default="http://ollama:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="mistral", env="OLLAMA_MODEL")
    ollama_timeout: int = Field(default=120, env="OLLAMA_TIMEOUT")
    
    # Redis Einstellungen
    redis_url: str = Field(default="redis://redis:6379", env="REDIS_URL")
    redis_ttl: int = Field(default=3600, env="REDIS_TTL")
    
    # Search Einstellungen
    min_relevance_score: float = Field(default=0.3, env="MIN_RELEVANCE_SCORE")
    max_search_results: int = Field(default=50, env="MAX_SEARCH_RESULTS")
    semantic_search_enabled: bool = Field(default=True, env="SEMANTIC_SEARCH_ENABLED")
    
    # OCR Einstellungen
    ocr_engine: str = Field(default="tesseract", env="OCR_ENGINE")
    tesseract_lang: str = Field(default="deu+eng", env="TESSERACT_LANG")
    ocr_dpi: int = Field(default=300, env="OCR_DPI")
    
    # Privacy Einstellungen
    enable_pii_removal: bool = Field(default=True, env="ENABLE_PII_REMOVAL")
    anonymize_metadata: bool = Field(default=True, env="ANONYMIZE_METADATA")
    
    # Performance Einstellungen
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    batch_size: int = Field(default=10, env="BATCH_SIZE")
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # Speicher Einstellungen
    data_path: str = Field(default="./data", env="DATA_PATH")
    index_path: str = Field(default="./indices", env="INDEX_PATH")
    log_path: str = Field(default="./logs", env="LOG_PATH")
    
    # N8N Einstellungen
    n8n_webhook_url: Optional[str] = Field(default=None, env="N8N_WEBHOOK_URL")
    n8n_api_key: Optional[str] = Field(default=None, env="N8N_API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def get_ollama_headers(self) -> dict:
        """Ollama API Headers"""
        return {
            "Content-Type": "application/json"
        }
    
    def get_redis_config(self) -> dict:
        """Redis Konfiguration"""
        return {
            "decode_responses": True,
            "max_connections": 50,
            "retry_on_timeout": True
        }

# Singleton Instance
api_config = APIConfig()
