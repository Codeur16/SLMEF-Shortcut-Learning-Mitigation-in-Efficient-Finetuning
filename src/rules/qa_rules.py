"""
Question answering rules
"""

import re
from .rule_engine import BaseRule

class LexicalBiasRule(BaseRule):
    """Rule for lexical bias in QA"""
    
    def __init__(self):
        super().__init__("lexical_bias")
    
    def applies_to(self, text: Any, **kwargs) -> bool:
        """Check for lexical overlap between question and passage"""
        if isinstance(text, dict) and "question" in text and "passage" in text:
            question = text["question"].lower()
            passage = text["passage"].lower()
            
            # Simple check: question words in passage
            question_words = set(re.findall(r'\b\w+\b', question))
            passage_words = set(re.findall(r'\b\w+\b', passage))
            
            overlap = len(question_words.intersection(passage_words))
            return overlap > len(question_words) * 0.5  # 50% overlap
        return False
    
    def get_suggested_label(self, text: Any, **kwargs) -> int:
        """High overlap suggests answer is True"""
        return 1

class QARuleEngine:
    """QA rule engine"""
    
    def __init__(self):
        self.rules = [LexicalBiasRule()]
    
    def apply(self, question: str, passage: str) -> dict:
        """Apply QA rules"""
        text = {"question": question, "passage": passage}
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