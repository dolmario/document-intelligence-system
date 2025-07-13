from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
import json

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
    references: List[str] = field(default_factory=list)  # doc_ids
    referenced_by: List[str] = field(default_factory=list)
    semantic_links: Dict[str, float] = field(default_factory=dict)  # doc_id: similarity
    
    # Zeitstempel (anonymisiert)
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: Optional[str] = None
    
    # Learning-Daten
    access_count: int = 0
    search_matches: List[str] = field(default_factory=list)  # anonymisierte Suchanfragen
    
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
        return json.dumps({
            'doc_id': self.doc_id,
            'title': self.title,
            'keywords': self.keywords,
            'categories': self.categories,
            'content_summary': ' '.join([s.get('content', '')[:200] 
                                       for s in self.sections[:3]]),
            'references': self.references,
            'semantic_links': self.semantic_links
        }, ensure_ascii=False, indent=2)
