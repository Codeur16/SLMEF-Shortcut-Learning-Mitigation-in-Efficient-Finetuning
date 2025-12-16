"""
Flan-T5 model with RG-PEFT support
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)

class FlanT5Model(nn.Module):
    """Flan-T5 model with PEFT support"""
    
    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__()
        
        self.config = config
        self.device = device
        self.model_type = "flan-t5"
        
        # Load model configuration
        model_name = config.get("model_name", "google/flan-t5-large")
        
        t5_config = T5Config.from_pretrained(
            model_name,
            decoder_start_token_id=0
        )
        
        # Load base model
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            config=t5_config
        )
        
        # Apply LoRA if enabled
        if config.get("lora", {}).get("enabled", False):
            lora_config = config["lora"]
            peft_config = LoraConfig(
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("lora_alpha", 32),
                lora_dropout=lora_config.get("lora_dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["q", "v"]),
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            self.model = get_peft_model(self.model, peft_config)
            logger.info(f"Applied LoRA to Flan-T5 model")
        
        # Move to device
        self.model.to(device)
        self.model.eval()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for sequence-to-sequence"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate text"""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )