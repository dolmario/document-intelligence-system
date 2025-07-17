import re
import hashlib
from typing import Dict, Tuple

class Anonymizer:
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(?:\+49|0049|0)\s*(?:\d{2,4}[\s-]?\d{4,}|\d{3,4}[\s-]?\d{6,})',
            'iban': r'\b[A-Z]{2}\d{2}\s?(?:[A-Z0-9]\s?){1,30}\b',
            'ip': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'name': r'\b(?:Herr|Frau|Dr\.|Prof\.)?\s*[A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+\b'
        }
        self.replacements = {}
    
    def anonymize(self, text: str) -> Tuple[str, Dict[str, str]]:
        self.replacements = {}
        
        for key, pattern in self.patterns.items():
            text = re.sub(pattern, lambda m: self._replace(m.group(), key), text, flags=re.IGNORECASE)
        
        return text, self.replacements
    
    def _replace(self, match: str, type: str) -> str:
        hash_val = hashlib.md5(match.encode()).hexdigest()[:8]
        replacement = f"[{type.upper()}_{hash_val}]"
        self.replacements[replacement] = match
        return replacement
    
    def deanonymize(self, text: str, replacements: Dict[str, str]) -> str:
        for placeholder, original in replacements.items():
            text = text.replace(placeholder, original)
        return text
