"""
NLI dataset handling
"""

from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class NLIDataset(Dataset):
    """Natural Language Inference dataset"""
    
    def __init__(self, config: dict, split: str = "validation", validation_type: str = "matched"):
        self.config = config
        self.split = split
        self.validation_type = validation_type  # 'matched' or 'mismatched'
    
        # Load datasets
        id_config = config.get("id_dataset", {})
        self.id_dataset = self._load_dataset(id_config, split, validation_type)
        
        ood_config = config.get("ood_dataset", {})
        if ood_config:
            self.ood_dataset = self._load_dataset(ood_config, split, validation_type)
        else:
            self.ood_dataset = None
        
        logger.info(f"Loaded NLI dataset: {len(self.id_dataset)} ID samples")
        if self.ood_dataset:
            logger.info(f"Loaded NLI OOD dataset: {len(self.ood_dataset)} samples")
    
    def _load_dataset(self, config: dict, split: str, validation_type: str = "matched"):
        """Load dataset from HuggingFace"""
        dataset_name = config.get("name", "nyu-mll/multi_nli")
        
        try:
            # Handle MNLI dataset which doesn't have a simple 'validation' split
            if dataset_name in ["nyu-mll/multi_nli", "multi_nli", "glue/mnli"]:
                if split == "validation":
                    if validation_type == "matched":
                        split = "validation_matched"
                    elif validation_type == "mismatched":
                        split = "validation_mismatched"
                    elif validation_type == "both":
                        # Load both and concatenate
                        matched = load_dataset(dataset_name, split="validation_matched")
                        mismatched = load_dataset(dataset_name, split="validation_mismatched")
                        return concatenate_datasets([matched, mismatched])
            
            # Try to load with trust_remote_code=False first
            try:
                return load_dataset(dataset_name, split=split)
            except (ValueError, TypeError):
                # Fall back to trust_remote_code=True if needed
                return load_dataset(dataset_name, split=split, trust_remote_code=True)
                
        except Exception as e:
            logger.warning(f"Failed to load dataset {dataset_name} (split: {split}): {str(e)}")
            logger.warning("Skipping this dataset.")
            return None
    
    def __len__(self):
        return len(self.id_dataset)
    
    def __getitem__(self, idx):
        item = self.id_dataset[idx]
        
        premise = item.get("premise", "")
        hypothesis = item.get("hypothesis", "")
        label = item.get("label", -1)
        
        return {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "text": f"Premise: {premise} Hypothesis: {hypothesis}"
        }
    
    def get_ood_sample(self, idx):
        """Get OOD sample"""
        if self.ood_dataset is None:
            return None
        
        item = self.ood_dataset[idx]
        premise = item.get("premise", "")
        hypothesis = item.get("hypothesis", "")
        label = item.get("label", -1)
        
        return {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "text": f"Premise: {premise} Hypothesis: {hypothesis}"
        }
    
    def tokenize(self, tokenizer, max_length=512):
        """Tokenize dataset"""
        def tokenize_function(examples):
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
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