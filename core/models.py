from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib
import json
from pathlib import Path

@dataclass
class DocumentIndex:
    """DSGVO-konforme Dokumenten-Index-Struktur"""
    
    # Identifikation (anonymisiert)
    doc_id: str  # SHA-256 Hash des Dateipfads
    doc_type: str  # pdf, docx, db_entry, etc.
    
    # Inhaltsdaten (keine PII)
    title: str
    content_hash: str  # Für Duplikaterkennung
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Strukturierte Inhalte
    sections: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    # Verknüpfungen
    references: List[str] = field(default_factory=list)
    referenced_by: List[str] = field(default_factory=list)
    semantic_links: Dict[str, float] = field(default_factory=dict)
    
    # Zeitstempel (anonymisiert)
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: Optional[str] = None
    
    # Learning-Daten
    access_count: int = 0
    search_matches: List[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Konvertiere zu durchsuchbarem Markdown"""
        md = f"# {self.title}\n\n"
        md += f"**ID**: {self.doc_id[:8]}...\n"
        md += f"**Type**: {self.doc_type}\n"
        md += f"**Categories**: {', '.join(self.categories)}\n"
        md += f"**Keywords**: {', '.join(self.keywords[:10])}\n\n"
        
        md += "## Content\n\n"
        for section in self.sections:
            md += f"### {section.get('title', 'Section')}\n"
            md += f"{section.get('content', '')}\n\n"
        
        if self.references:
            md += "## References\n"
            for ref in self.references:
                md += f"- [{ref[:8]}...](#{ref})\n"
        
        return md
    
    def to_json(self) -> str:
        """Konvertiere zu JSON für schnelle Suche"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DocumentIndex':
        """Erstelle DocumentIndex aus JSON"""
        data = json.loads(json_str)
        return cls(**data)
    
    def save(self, base_path: Path):
        """Speichere Index als Markdown und JSON"""
        # Erstelle Verzeichnisse falls nicht vorhanden
        md_path = base_path / "markdown" / f"{self.doc_id}.md"
        json_path = base_path / "json" / f"{self.doc_id}.json"
        
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Speichere Dateien
        md_path.write_text(self.to_markdown(), encoding='utf-8')
        json_path.write_text(self.to_json(), encoding='utf-8')

@dataclass
class ProcessingTask:
    """Aufgabe für die Verarbeitungs-Pipeline"""
    task_id: str
    file_path: str
    task_type: str  # 'ocr', 'index', 'update'
    priority: int = 5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = 'pending'  # pending, processing, completed, failed
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
