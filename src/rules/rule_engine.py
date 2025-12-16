"""
Base rule engine for detecting shortcuts
"""
from typing import Any
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseRule(ABC):
    """Base class for all rules"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def applies_to(self, text: Any, **kwargs) -> bool:
        """Check if rule applies to text"""
        pass
    
    @abstractmethod
    def get_suggested_label(self, text: Any, **kwargs) -> Any:
        """Get suggested label if rule applies"""
        pass

class RuleEngine:
    """Engine for applying multiple rules"""
    
    def __init__(self, rules: List[BaseRule]):
        self.rules = rules
    
    def apply(self, text: Any, **kwargs) -> Dict[str, Any]:
        """Apply all rules to text"""
        results = {
            "activations": [],
            "suggestions": [],
            "active_rules": []
        }
        
        for rule in self.rules:
            if rule.applies_to(text, **kwargs):
                results["activations"].append(True)
                results["suggestions"].append(rule.get_suggested_label(text, **kwargs))
                results["active_rules"].append(rule.name)
            else:
                results["activations"].append(False)
                results["suggestions"].append(None)
        
        return results
    
    def compute_rule_loss(self, model_logits, rule_results, lambda_reg=0.1):
        """Compute rule-guided loss"""
        import torch
        
        active_rules = rule_results["activations"]
        suggestions = rule_results["suggestions"]
        
        if not any(active_rules):
            return torch.tensor(0.0)
        
        # Simple regularization: penalize confidence for rule-activated examples
        loss = 0.0
        for i, active in enumerate(active_rules):
            if active and suggestions[i] is not None:
                # Penalize high probability for suggested (potentially wrong) label
                prob = torch.softmax(model_logits, dim=-1)[:, suggestions[i]]
                loss += torch.mean(prob)
        
        return lambda_reg * loss / sum(active_rules)