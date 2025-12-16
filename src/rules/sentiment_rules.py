"""
Sentiment analysis rules
"""
from typing import Any
from .rule_engine import BaseRule

class SentimentLexiconRule(BaseRule):
    """Rule based on sentiment lexicon"""
    
    def __init__(self):
        super().__init__("sentiment_lexicon")
        self.positive_words = {"good", "great", "excellent", "wonderful", "amazing", "love"}
        self.negative_words = {"bad", "terrible", "awful", "horrible", "worst", "hate"}
    
    def applies_to(self, text: Any, **kwargs) -> bool:
        """Check if text contains sentiment words"""
        if isinstance(text, str):
            text_lower = text.lower()
            has_pos = any(word in text_lower for word in self.positive_words)
            has_neg = any(word in text_lower for word in self.negative_words)
            return has_pos or has_neg
        return False
    
    def get_suggested_label(self, text: Any, **kwargs) -> int:
        """Determine suggested label based on lexicon"""
        if isinstance(text, str):
            text_lower = text.lower()
            pos_count = sum(1 for word in self.positive_words if word in text_lower)
            neg_count = sum(1 for word in self.negative_words if word in text_lower)
            return 1 if pos_count > neg_count else 0
        return 0

class NegationRule(BaseRule):
    """Rule for negation patterns"""
    
    def __init__(self):
        super().__init__("negation")
        self.negation_words = {"not", "no", "never", "none", "nothing"}
    
    def applies_to(self, text: Any, **kwargs) -> bool:
        """Check for negation"""
        if isinstance(text, str):
            words = text.lower().split()
            return any(word in self.negation_words for word in words)
        return False
    
    def get_suggested_label(self, text: Any, **kwargs) -> int:
        """Negation often suggests negative sentiment"""
        return 0

class SentimentRuleEngine:
    """Sentiment analysis rule engine"""
    
    def __init__(self):
        self.rules = [
            SentimentLexiconRule(),
            NegationRule()
        ]
    
    def apply(self, text: str) -> dict:
        """Apply sentiment rules"""
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