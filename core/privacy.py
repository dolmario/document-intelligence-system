import re
import hashlib
from typing import Dict, List, Any
import spacy
from datetime import datetime

class PrivacyManager:
    """Manager für DSGVO-konforme Datenverarbeitung"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("de_core_news_sm")
        except:
            print("Spacy model nicht gefunden. Installiere mit: python -m spacy download de_core_news_sm")
            self.nlp = None
            
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+49|0)[1-9]\d{1,14}\b',
            'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b',
            'personal_id': r'\b\d{8,12}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'credit_card': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b',
        }
        
        self.sensitive_orgs = [
            'krankenhaus', 'klinik', 'praxis', 'versicherung',
            'bank', 'sparkasse', 'finanzamt', 'gericht'
        ]
    
    def anonymize_text(self, text: str) -> str:
        """Entferne oder anonymisiere PII aus Text"""
        if not text:
            return text
            
        # Ersetze Pattern-basierte PII
        for pii_type, pattern in self.pii_patterns.items():
            text = re.sub(pattern, f'[{pii_type.upper()}]', text, flags=re.IGNORECASE)
        
        # NER-basierte Anonymisierung wenn Spacy verfügbar
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'PER':
                    text = text.replace(ent.text, '[PERSON]')
                elif ent.label_ == 'LOC' and len(ent.text) > 3:
                    text = text.replace(ent.text, '[LOCATION]')
                elif ent.label_ == 'ORG' and self._is_sensitive_org(ent.text):
                    text = text.replace(ent.text, '[ORGANIZATION]')
        
        return text
    
    def _is_sensitive_org(self, org_name: str) -> bool:
        """Prüfe ob Organisation sensibel ist"""
        org_lower = org_name.lower()
        return any(sens in org_lower for sens in self.sensitive_orgs)
    
    def anonymize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymisiere Metadaten"""
        anonymized = metadata.copy()
        
        # Liste von Feldern die anonymisiert werden sollen
        fields_to_hash = ['author', 'creator', 'user', 'owner', 'email']
        fields_to_remove = ['gps', 'location', 'address', 'ip']
        
        for field in fields_to_hash:
            if field in anonymized:
                anonymized[field] = self.hash_value(str(anonymized[field]))
        
        for field in fields_to_remove:
            anonymized.pop(field, None)
        
        # Zeitstempel normalisieren (nur Datum, keine genaue Zeit)
        for field in ['created', 'modified', 'accessed']:
            if field in anonymized and isinstance(anonymized[field], str):
                try:
                    dt = datetime.fromisoformat(anonymized[field])
                    anonymized[field] = dt.date().isoformat()
                except:
                    pass
        
        return anonymized
    
    def hash_value(self, value: str) -> str:
        """Erstelle anonymen Hash eines Wertes"""
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def generate_anonymous_id(self, filepath: str) -> str:
        """Generiere anonyme ID aus Dateipfad"""
        return hashlib.sha256(filepath.encode()).hexdigest()
