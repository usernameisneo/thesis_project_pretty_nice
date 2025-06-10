from typing import List, Dict

class CitationPatterns:
    def __init__(self):
        self.citation_patterns = [
            r'\b\d+(\.\d+)?%\b',
            r'\b\d+(\.\d+)?\s*(million|billion|thousand)\b',
            r'\baccording to\b',
            r'\bresearch shows?\b',
            r'\bstudies (show|indicate|suggest)\b',
            r'\bdata (shows?|indicates?|suggests?)\b',
            r'\bit is (known|established|proven) that\b',
            r'\bevidence suggests?\b',
            r'\bfindings (show|indicate|reveal)\b',
            r'\banalysis (shows?|reveals?|indicates?)\b',
            r'\btheory (states?|suggests?|proposes?)\b',
            r'\bmodel (predicts?|suggests?|shows?)\b',
            r'\bframework (indicates?|suggests?)\b',
            r'\b(more|less|higher|lower|greater|smaller) than\b',
            r'\bcompared to\b',
            r'\bin contrast to\b',
            r'\bunlike\b',
            r'\b(causes?|leads? to|results? in)\b',
            r'\b(due to|because of|as a result of)\b',
            r'\b(influences?|affects?|impacts?)\b'
        ]

        self.no_citation_patterns = [
            r'\bi think\b',
            r'\bin my opinion\b',
            r'\bit seems\b',
            r'\bperhaps\b',
            r'\bmight be\b',
            r'\bcould be\b',
            r'\bthis thesis\b',
            r'\bthis study\b',
            r'\bthis research\b'
        ]

        self.academic_signals: Dict[str, List[str]] = {
            'factual': ['established', 'proven', 'demonstrated', 'confirmed'],
            'statistical': ['percentage', 'rate', 'frequency', 'proportion'],
            'research': ['study', 'research', 'investigation', 'analysis'],
            'theoretical': ['theory', 'model', 'framework', 'concept'],
            'methodological': ['method', 'approach', 'technique', 'procedure'],
            'comparative': ['compared', 'versus', 'relative', 'contrast'],
            'causal': ['cause', 'effect', 'influence', 'impact', 'result'],
            'definitional': ['defined', 'definition', 'refers to', 'means'],
            'evaluative': ['effective', 'successful', 'important', 'significant'],
            'historical': ['historically', 'traditionally', 'previously', 'past']
        }


