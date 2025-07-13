# index_generator.py
import re
from typing import List, Dict, Any
import spacy
from collections import Counter
import hashlib

class DSGVOCompliantIndexer:
    def __init__(self):
        self.nlp = spacy.load("de_core_news_sm")
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+49|0)[1-9]\d{1,14}\b',
            'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b',
            'personal_id': r'\b\d{8,12}\b',  # Personalausweis, etc.
        }
        
    def generate_index(self, content: str, metadata: dict) -> DocumentIndex:
        """Erstelle DSGVO-konformen Index"""
        
        # 1. Bereinige PII
        clean_content = self.remove_pii(content)
        
        # 2. Extrahiere Struktur
        sections = self.extract_sections(clean_content)
        
        # 3. Generiere Keywords
        keywords = self.extract_keywords(clean_content)
        
        # 4. Erkenne Referenzen
        references = self.detect_references(clean_content)
        
        # 5. Erstelle anonyme ID
        doc_id = self.generate_anonymous_id(metadata.get('path', ''))
        
        # 6. Kategorisiere
        categories = self.auto_categorize(clean_content, keywords)
        
        return DocumentIndex(
            doc_id=doc_id,
            doc_type=metadata.get('type', 'unknown'),
            title=self.extract_title(clean_content, metadata),
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            keywords=keywords,
            categories=categories,
            sections=sections,
            metadata=self.anonymize_metadata(metadata),
            references=references
        )
    
    def remove_pii(self, text: str) -> str:
        """Entferne personenbezogene Daten"""
        # Ersetze E-Mails
        text = re.sub(self.pii_patterns['email'], '[EMAIL]', text)
        
        # Ersetze Telefonnummern
        text = re.sub(self.pii_patterns['phone'], '[PHONE]', text)
        
        # Ersetze IBANs
        text = re.sub(self.pii_patterns['iban'], '[IBAN]', text)
        
        # Nutze NER für Namen
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['PER', 'LOC', 'ORG']:
                if ent.label_ == 'PER':
                    text = text.replace(ent.text, '[PERSON]')
                elif ent.label_ == 'LOC':
                    text = text.replace(ent.text, '[LOCATION]')
                elif ent.label_ == 'ORG' and self.is_sensitive_org(ent.text):
                    text = text.replace(ent.text, '[ORGANIZATION]')
        
        return text
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """Extrahiere relevante Keywords"""
        doc = self.nlp(text)
        
        # Filtere relevante Tokens
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct 
                 and len(token.text) > 2]
        
        # Zähle Häufigkeit
        word_freq = Counter(tokens)
        
        # Extrahiere Noun Phrases
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                       if len(chunk.text.split()) <= 3]
        
        # Kombiniere und ranke
        keywords = list(word_freq.most_common(top_n))
        keywords.extend([(np, 1) for np in noun_phrases[:10]])
        
        return [kw[0] for kw in sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]]
    
    def detect_references(self, text: str) -> List[str]:
        """Erkenne Verweise auf andere Dokumente"""
        references = []
        
        # Muster für Dokumentverweise
        patterns = [
            r'siehe auch[:\s]+([^\.]+)',
            r'vgl\.[:\s]+([^\.]+)',
            r'basiert auf[:\s]+([^\.]+)',
            r'Fortsetzung von[:\s]+([^\.]+)',
            r'Anhang[:\s]+([^\.]+)',
            r'Dokument[:\s]+([^\.]+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ref_text = match.group(1).strip()
                # Generiere Hash für Referenz
                ref_id = hashlib.sha256(ref_text.encode()).hexdigest()
                references.append(ref_id)
        
        return list(set(references))
