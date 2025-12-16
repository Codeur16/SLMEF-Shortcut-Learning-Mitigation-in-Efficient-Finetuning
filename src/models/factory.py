"""
Model factory for creating different model types
"""

import torch
import yaml
import os
from typing import Dict, Any, Optional

from .bert_model import BERTModel
from .roberta_model import RoBERTaModel
from .flant5_model import FlanT5Model
from .llama_model import LLaMAModel
from .mistral_model import MistralModel

class ModelFactory:
    """Factory for creating model instances"""
    
    MODEL_CLASSES = {
        "bert": BERTModel,
        "roberta": RoBERTaModel,
        "flan-t5": FlanT5Model,
        "llama": LLaMAModel,
        "mistral": MistralModel,
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        """Create a model instance"""
        if model_type not in cls.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls.MODEL_CLASSES[model_type]
        return model_class(config, device)
    
    @classmethod
    def create_from_config(
        cls,
        model_name: str,
        config_path: str,
        device: str = "cuda"
    ):
        """Create model from configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls.create_model(model_name, config, device)
    
    @classmethod
    def get_tokenizer(cls, model_type: str, model_name: str):
        """Get tokenizer for a model type"""
        from transformers import AutoTokenizer
        
        tokenizer_kwargs = {}
        
        if model_type in ["llama", "mistral"]:
            tokenizer_kwargs = {
                "padding_side": "left",
                "truncation_side": "left"
            }
        # Configure tokenizer for FLAN-T5 model with right-side padding and truncation
        # to maintain consistency with the model's left-to-right processing
        elif model_type == "flan-t5":
            tokenizer_kwargs = {
                "padding_side": "right",
                "tokenizer_class": "T5Tokenizer",
                "use_fast": True,
                "truncation_side": "right"
            }
        
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        if tokenizer.pad_token is None:
            if model_type in ["llama", "mistral"]:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        return tokenizer