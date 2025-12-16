"""
RoBERTa model with RG-PEFT support
"""

import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaConfig
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__) # Logger for RoBERT model class

class RoBERTaModel(nn.Module):
    """
    RoBERTa model with PEFT support

    This class represents a RoBERTa model with PEFT support.
    It initializes the model with the given configuration and device.
    The forward method is used for the forward pass of the model.
    The predict method is used to generate predictions.
    """
    
    def __init__(self, config: dict, device: str = "cuda"):
        """
        Initialize the RoBERTa model with PEFT support.

        Args:
            config (dict): The configuration for the model.
            device (str, optional): The device to use. Defaults to "cuda".
        """
        super().__init__()
        
        self.config = config
        self.device = device
        self.model_type = "roberta"
        
        # Load model configuration
        model_name = config.get("model_name", "roberta-large")
        num_labels = config.get("num_labels", 2)
        
        roberta_config = RobertaConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        # Load base model
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            config=roberta_config
        )
        
        # Apply LoRA if enabled
        if config.get("lora", {}).get("enabled", False):
            lora_config = config["lora"]
            peft_config = LoraConfig(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 32),
                lora_dropout=lora_config.get("lora_dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["query", "value"]),
                bias="none",
                task_type="SEQ_CLS"
            )
            self.model = get_peft_model(self.model, peft_config)
            logger.info(f"Applied LoRA to RoBERTa model")
        
        # Move to device
        self.model.to(device)
        self.model.eval()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Perform the forward pass of the model.

        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (torch.Tensor, optional): The attention mask. Defaults to None.
            labels (torch.Tensor, optional): The labels. Defaults to None.

        Returns:
            torch.Tensor: The output of the model.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def predict(self, input_ids, attention_mask=None):
        """
        Generate predictions.

        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (torch.Tensor, optional): The attention mask. Defaults to None.

        Returns:
            torch.Tensor: The logits of the model.
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs.logits