# models/base_model.py

import os
from utils.config_loader import load_yaml_config

class BaseModel:
    """
    Base abstract model class.
    Every model (FlanT5, LLaMA, BERT, Mistral, GPT) will inherit from this class.
    
    The goal is to provide a unified interface for:
      - loading the model
      - preparing it for training (LoRA, QLoRA)
      - fine-tuning
      - inference
      - saving
    """
    
    def __init__(self, model_name, exp_type,  checkpoint=None ):
        self.config = load_yaml_config(model_name)
        # Some model implementations expect `self.cfg` — keep alias for compatibility
        self.cfg = self.config
        self.model_name = model_name
        self.exp_type = exp_type
        self.checkpoint = checkpoint
        self.model = None
        self.tokenizer = None
    def load(self):
        """Load base HF model + tokenizer."""
        raise NotImplementedError("load() must be implemented in subclasses.")
    
    def prepare_for_training(self):
        """Apply LoRA/QLoRA/Adapters or other configurations."""
        raise NotImplementedError("prepare_for_training() must be implemented in subclasses.")
    
    def train(self, train_dataset, val_dataset, save_dir):
        """Each model must override this method."""
        raise NotImplementedError("train() must be implemented in subclasses.")
    
    def inference(self, dataset, output_file):
        """Run inference over a dataset."""
        raise NotImplementedError("inference() must be implemented in subclasses.")
    
    def save(self, save_dir):
        """Save the model after training."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
