"""
Question answering dataset
"""

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class QADataset(Dataset):
    """Question answering dataset"""
    
    def __init__(self, config: dict, split: str = "validation"):
        self.config = config
        self.split = split
        
        # Load dataset
        id_config = config.get("id_dataset", {})
        self.dataset = self._load_dataset(id_config, split)
        
        logger.info(f"Loaded QA dataset: {len(self.dataset)} samples")
    
    def _load_dataset(self, config: dict, split: str):
        """Load dataset from HuggingFace"""
        dataset_name = config.get("name", "google/boolq")
        return load_dataset(dataset_name, split=split)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        question = item.get("question", "")
        passage = item.get("passage", "")
        label = 1 if item.get("answer", False) else 0
        
        return {
            "question": question,
            "passage": passage,
            "label": label,
            "text": f"Question: {question} Passage: {passage}"
        }
    
    def tokenize(self, tokenizer, max_length=512):
        """Tokenize dataset"""
        def tokenize_function(examples):
            return tokenizer(
                examples["question"],
                examples["passage"],
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