"""
LLaMA model with QLoRA support
"""

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)

class LLaMAModel(nn.Module):
    """LLaMA model with QLoRA support (4-bit)"""
    
    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__()
        
        self.config = config
        self.device = device
        self.model_type = "llama"
        
        # Load model configuration
        model_name = config.get("model_name", "meta-llama/Llama-2-7b-hf")
        
        # Configure 4-bit quantization for QLoRA
        bnb_config = None
        if config.get("qlora", {}).get("enabled", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        
        # Load base model with quantization
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        if bnb_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply QLoRA if enabled
        if config.get("qlora", {}).get("enabled", False):
            qlora_config = config["qlora"]
            peft_config = LoraConfig(
                r=qlora_config.get("r", 16),
                lora_alpha=qlora_config.get("lora_alpha", 32),
                lora_dropout=qlora_config.get("lora_dropout", 0.1),
                target_modules=qlora_config.get("target_modules", ["q_proj", "v_proj"]),
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, peft_config)
            logger.info(f"Applied QLoRA to LLaMA model (4-bit)")
        
        self.model.eval()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for causal LM"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate text"""
        generation_config = self.config.get("generation", {})
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=generation_config.get("max_new_tokens", 128),
            temperature=generation_config.get("temperature", 0.7),
            do_sample=True,
            **kwargs
        )