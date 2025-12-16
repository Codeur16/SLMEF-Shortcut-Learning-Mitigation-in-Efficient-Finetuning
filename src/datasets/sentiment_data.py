"""
Sentiment analysis dataset
"""

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    """Sentiment analysis dataset"""
    
    def __init__(self, config: dict, split: str = "validation"):
        self.config = config
        self.split = split
        
        # Load dataset
        id_config = config.get("id_dataset", {})
        self.dataset = self._load_dataset(id_config, split)
        
        logger.info(f"Loaded sentiment dataset: {len(self.dataset)} samples")
    
    def _load_dataset(self, config: dict, split: str):
        """Load dataset from HuggingFace"""
        dataset_name = config.get("name", "stanfordnlp/sst2")
        return load_dataset(dataset_name, split=split)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        text = item.get("sentence", "")
        label = item.get("label", -1)
        
        return {
            "text": text,
            "label": label
        }
    
    def tokenize(self, tokenizer, max_length=512):
        """Tokenize dataset"""
        def tokenize_function(examples):
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        
        tokenized = self.dataset.map(
            tokenize_function,
            batched=True
        )
        
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        return tokenized, None