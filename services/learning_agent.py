# learning_agent.py
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta

class LinkLearningAgent:
    def __init__(self):
        self.co_access_matrix = defaultdict(lambda: defaultdict(int))
        self.search_patterns = defaultdict(list)
        self.manual_links = defaultdict(set)
        
    def learn_from_search(self, query: str, clicked_results: List[str]):
        """Lerne aus Suchmustern"""
        # Speichere Suchmuster
        pattern = {
            'query': self.anonymize_query(query),
            'results': clicked_results,
            'timestamp': datetime.now().isoformat()
        }
        
        for doc_id in clicked_results:
            self.search_patterns[doc_id].append(pattern)
        
        # Aktualisiere Co-Access Matrix
        for i, doc1 in enumerate(clicked_results):
            for doc2 in clicked_results[i+1:]:
                self.co_access_matrix[doc1][doc2] += 1
                self.co_access_matrix[doc2][doc1] += 1
    
    def suggest_links(self, doc_id: str, threshold: float = 0.3) -> Dict[str, float]:
        """Schlage Verknüpfungen basierend auf Mustern vor"""
        suggestions = {}
        
        # 1. Co-Access basierte Vorschläge
        if doc_id in self.co_access_matrix:
            total_accesses = sum(self.co_access_matrix[doc_id].values())
            
            for linked_doc, count in self.co_access_matrix[doc_id].items():
                confidence = count / total_accesses
                if confidence > threshold:
                    suggestions[linked_doc] = confidence
        
        # 2. Suchmuster-basierte Vorschläge
        doc_queries = [p['query'] for p in self.search_patterns.get(doc_id, [])]
        
        for other_doc, patterns in self.search_patterns.items():
            if other_doc != doc_id:
                other_queries = [p['query'] for p in patterns]
                
                # Berechne Query-Überlappung
                overlap = len(set(doc_queries) & set(other_queries))
                if overlap > 0:
                    overlap_score = overlap / max(len(doc_queries), len(other_queries))
                    if overlap_score > threshold:
                        if other_doc in suggestions:
                            suggestions[other_doc] = max(suggestions[other_doc], overlap_score)
                        else:
                            suggestions[other_doc] = overlap_score
        
        # 3. Manuelle Verknüpfungen (höchste Priorität)
        for linked_doc in self.manual_links.get(doc_id, set()):
            suggestions[linked_doc] = 1.0
        
        return suggestions
    
    def add_manual_link(self, doc1: str, doc2: str, bidirectional: bool = True):
        """Füge manuelle Verknüpfung hinzu (z.B. für aufeinander aufbauende Arbeiten)"""
        self.manual_links[doc1].add(doc2)
        if bidirectional:
            self.manual_links[doc2].add(doc1)
    
    def detect_sequential_documents(self, indices: Dict[str, DocumentIndex]):
        """Erkenne aufeinander aufbauende Dokumente"""
        for doc_id, index in indices.items():
            # Suche nach Hinweisen auf Fortsetzungen
            content = ' '.join([s['content'] for s in index.sections])
            
            sequential_patterns = [
                r'Teil (\d+) von (\d+)',
                r'Fortsetzung folgt',
                r'siehe Teil (\d+)',
                r'basiert auf (.+)',
                r'Vorarbeit[:\s]+(.+)',
            ]
            
            for pattern in sequential_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Versuche verwandte Dokumente zu finden
                    # ... Implementierung der Logik
                    pass
