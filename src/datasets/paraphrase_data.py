"""
Paraphrase detection dataset
"""

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class ParaphraseDataset(Dataset):
    """Paraphrase detection dataset"""
    
    def __init__(self, config: dict, split: str = "validation"):
        self.config = config
        self.split = split
        
        # Load datasets
        id_config = config.get("id_dataset", {})
        self.id_dataset = self._load_dataset(id_config, split)
        
        ood_config = config.get("ood_dataset", {})
        if ood_config:
            self.ood_dataset = self._load_dataset(ood_config, split)
        else:
            self.ood_dataset = None
        
        logger.info(f"Loaded paraphrase dataset: {len(self.id_dataset)} ID samples")
        if self.ood_dataset:
            logger.info(f"Loaded paraphrase OOD dataset: {len(self.ood_dataset)} samples")
    
    def _load_dataset(self, config: dict, split: str):
        """Load dataset from HuggingFace"""
        dataset_name = config.get("name", "SetFit/mrpc")
        config_name = config.get("config", None)
        
        if config_name:
            return load_dataset(dataset_name, config_name, split=split)
        return load_dataset(dataset_name, split=split)
    
    def __len__(self):
        return len(self.id_dataset)
    
    def __getitem__(self, idx):
        item = self.id_dataset[idx]
        
        sentence1 = item.get("sentence1", "")
        sentence2 = item.get("sentence2", "")
        label = item.get("label", -1)
        
        return {
            "sentence1": sentence1,
            "sentence2": sentence2,
            "label": label,
            "text": f"Sentence 1: {sentence1} Sentence 2: {sentence2}"
        }
    
    def get_ood_sample(self, idx):
        """Get OOD sample"""
        if self.ood_dataset is None:
            return None
        
        item = self.ood_dataset[idx]
        sentence1 = item.get("sentence1", "")
        sentence2 = item.get("sentence2", "")
        label = item.get("label", -1)
        
        return {
            "sentence1": sentence1,
            "sentence2": sentence2,
            "label": label,
            "text": f"Sentence 1: {sentence1} Sentence 2: {sentence2}"
        }
    
    def tokenize(self, tokenizer, max_length=512):
        """Tokenize dataset"""
        def tokenize_function(examples):
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        
        tokenized_id = self.id_dataset.map(
            tokenize_function,
            batched=True
        )
        
        tokenized_id.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        if self.ood_dataset:
            tokenized_ood = self.ood_dataset.map(
                tokenize_function,
                batched=True
            )
            tokenized_ood.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            return tokenized_id, tokenized_ood
        
        return tokenized_id, None