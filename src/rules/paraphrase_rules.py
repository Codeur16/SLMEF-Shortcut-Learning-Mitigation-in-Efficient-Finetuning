"""
Paraphrase detection rules
"""
from typing import Any
import re
from .rule_engine import BaseRule

class SurfaceSimilarityRule(BaseRule):
    """Rule for surface similarity in paraphrase detection"""
    
    def __init__(self, threshold: float = 0.8):
        super().__init__("surface_similarity")
        self.threshold = threshold
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between texts"""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def applies_to(self, text: Any, **kwargs) -> bool:
        """Check for surface similarity"""
        if isinstance(text, dict) and "sentence1" in text and "sentence2" in text:
            similarity = self._jaccard_similarity(text["sentence1"], text["sentence2"])
            return similarity > self.threshold
        return False
    
    def get_suggested_label(self, text: Any, **kwargs) -> int:
        """High surface similarity suggests paraphrase (label 1)"""
        return 1

class ParaphraseRuleEngine:
    """Paraphrase detection rule engine"""
    
    def __init__(self):
        self.rules = [SurfaceSimilarityRule(threshold=0.7)]
    
    def apply(self, sentence1: str, sentence2: str) -> dict:
        """Apply paraphrase rules"""
        text = {"sentence1": sentence1, "sentence2": sentence2}
        activations = []
        suggestions = []
        
        for rule in self.rules:
            if rule.applies_to(text):
                activations.append(True)
                suggestions.append(rule.get_suggested_label(text))
            else:
                activations.append(False)
                suggestions.append(None)
        
        return {
            "activations": activations,
            "suggestions": suggestions,
            "num_active": sum(activations)
        }