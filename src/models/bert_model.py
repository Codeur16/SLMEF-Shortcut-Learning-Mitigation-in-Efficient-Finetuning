"""
BERT model with RG-PEFT support
"""

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)  # Logger for BERTModel class

class BERTModel(nn.Module):
    """BERT model with PEFT support"""
    
    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__()
        
        self.config = config
        self.device = device
        self.model_type = "bert"
        
        # Load model configuration
        model_name = config.get("model_name", "bert-base-uncased")
        num_labels = config.get("num_labels", 2)
        
        bert_config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        # Load base model
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            config=bert_config
        )
        
        # Apply LoRA if enabled
        if config.get("lora", {}).get("enabled", False):
            lora_config = config["lora"]
            peft_config = LoraConfig(
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("lora_alpha", 32),
                lora_dropout=lora_config.get("lora_dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["query", "value"]),
                bias="none",
                task_type="SEQ_CLS"
            )
            self.model = get_peft_model(self.model, peft_config)
            logger.info(f"Applied LoRA to BERT model")
        
        # Move to device
        self.model.to(device)
        self.model.eval()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def predict(self, input_ids, attention_mask=None):
        """Generate predictions"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs.logits
    
    def save_pretrained(self, path: str):
        """Save model"""
        self.model.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda"):
        """Load model from path"""
        from transformers import BertForSequenceClassification
        
        model = BertForSequenceClassification.from_pretrained(path)
        instance = cls({}, device)
        instance.model = model.to(device)
        return instance