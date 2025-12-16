"""
NLI-specific rules for shortcut detection
"""
from typing import Any
import re
from .rule_engine import BaseRule


class LexicalOverlapRule(BaseRule):
    """Rule for lexical overlap shortcut in NLI"""
    
    def __init__(self, threshold: float = 0.7):
        super().__init__("lexical_overlap")
        self.threshold = threshold
    
    def _token_overlap(self, text1: str, text2: str) -> float:
        """Compute token overlap ratio"""
        tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
        tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        overlap = len(tokens1.intersection(tokens2))
        return overlap / len(tokens1)
    
    def applies_to(self, text: Any, **kwargs) -> bool:
        """Check for lexical overlap"""
        if isinstance(text, dict) and "premise" in text and "hypothesis" in text:
            premise = text["premise"]
            hypothesis = text["hypothesis"]
            overlap = self._token_overlap(premise, hypothesis)
            return overlap > self.threshold
        return False
    
    def get_suggested_label(self, text: Any, **kwargs) -> int:
        """High overlap suggests entailment (label 0)"""
        return 0

class SubsequenceRule(BaseRule):
    """Rule for subsequence shortcut in NLI"""
    
    def __init__(self):
        super().__init__("subsequence")
    
    def applies_to(self, text: Any, **kwargs) -> bool:
        """Check if hypothesis is subsequence of premise"""
        if isinstance(text, dict) and "premise" in text and "hypothesis" in text:
            premise = text["premise"].lower()
            hypothesis = text["hypothesis"].lower()
            return hypothesis in premise
        return False
    
    def get_suggested_label(self, text: Any, **kwargs) -> int:
        """Subsequence suggests entailment"""
        return 0

class NLIRuleEngine:
    """NLI-specific rule engine"""
    
    def __init__(self):
        self.rules = [
            LexicalOverlapRule(threshold=0.6),
            SubsequenceRule()
        ]
    
    def apply(self, premise: str, hypothesis: str) -> dict:
        """Apply NLI rules"""
        text = {"premise": premise, "hypothesis": hypothesis}
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