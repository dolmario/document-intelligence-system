import re
import hashlib
from typing import List, Dict
from collections import Counter
from pathlib import Path

import spacy

from core.models import DocumentIndex  # Diese Klasse muss im Projekt definiert sein


class DSGVOCompliantIndexer:
    def __init__(self):
        self.nlp = spacy.load("de_core_news_sm")
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+49|0)[1-9]\d{1,14}\b',
            'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b',
            'personal_id': r'\b\d{8,12}\b',
        }

    def generate_index(self, content: str, metadata: dict) -> DocumentIndex:
        clean_content = self.remove_pii(content)
        sections = self.extract_sections(clean_content)
        keywords = self.extract_keywords(clean_content)
        references = self.detect_references(clean_content)
        doc_id = self.generate_anonymous_id(metadata.get('path', ''))
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
        text = re.sub(self.pii_patterns['email'], '[EMAIL]', text)
        text = re.sub(self.pii_patterns['phone'], '[PHONE]', text)
        text = re.sub(self.pii_patterns['iban'], '[IBAN]', text)
        text = re.sub(self.pii_patterns['personal_id'], '[ID]', text)

        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'PER':
                text = text.replace(ent.text, '[PERSON]')
            elif ent.label_ == 'LOC':
                text = text.replace(ent.text, '[LOCATION]')
            elif ent.label_ == 'ORG' and self.is_sensitive_org(ent.text):
                text = text.replace(ent.text, '[ORGANIZATION]')

        return text

    def extract_sections(self, text: str) -> List[Dict]:
        sections = []
        paragraphs = text.split('\n\n')
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 50:
                sections.append({
                    'title': f'Abschnitt {i+1}',
                    'content': para.strip()[:1000]
                })
        return sections[:10]

    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc
                  if not token.is_stop and not token.is_punct and len(token.text) > 2]
        word_freq = Counter(tokens)
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]

        keywords = list(word_freq.most_common(top_n))
        keywords.extend([(np, 1) for np in noun_phrases[:10]])
        return [kw[0] for kw in sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]]

    def detect_references(self, text: str) -> List[str]:
        references = []
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
                ref_id = hashlib.sha256(ref_text.encode()).hexdigest()
                references.append(ref_id)
        return list(set(references))

    def generate_anonymous_id(self, filepath: str) -> str:
        return hashlib.sha256(filepath.encode()).hexdigest()

    def extract_title(self, content: str, metadata: dict) -> str:
        lines = content.split('\n')
        for line in lines[:5]:
            if 10 < len(line) < 100:
                return line.strip()
        return Path(metadata.get('path', 'Unbekannt')).stem

    def auto_categorize(self, content: str, keywords: List[str]) -> List[str]:
        categories = []
        category_keywords = {
            'Rechnung': ['rechnung', 'invoice', 'betrag', 'mwst'],
            'Vertrag': ['vertrag', 'contract', 'unterschrift', 'vereinbarung'],
            'Bericht': ['bericht', 'report', 'analyse', 'zusammenfassung'],
            'Korrespondenz': ['sehr geehrte', 'mit freundlichen', 'brief', 'email'],
            'Technisch': ['code', 'funktion', 'system', 'datenbank', 'api']
        }
        content_lower = content.lower()
        for category, cat_keywords in category_keywords.items():
            if any(kw in content_lower for kw in cat_keywords):
                categories.append(category)
        return categories[:3]

    def is_sensitive_org(self, org_text: str) -> bool:
        sensitive_terms = [
            'krankenhaus', 'klinik', 'versicherung', 'bank',
            'gericht', 'polizei', 'finanzamt'
        ]
        org_lower = org_text.lower()
        return any(term in org_lower for term in sensitive_terms)

    def anonymize_metadata(self, metadata: dict) -> dict:
        safe_metadata = {}
        safe_fields = ['type', 'size', 'page_count']
        for field in safe_fields:
            if field in metadata:
                safe_metadata[field] = metadata[field]
        return safe_metadata
