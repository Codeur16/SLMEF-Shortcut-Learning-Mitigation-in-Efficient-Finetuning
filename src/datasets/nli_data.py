"""
NLI dataset handling
"""

from datasets import load_dataset, concatenate_datasets
import os
import torch
from torch.utils.data import Dataset
import logging
from typing import Optional

from .hans_dataset import HANSDataset

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
    
    def _load_dataset(self, config: dict, split: str, validation_type: str = "matched") -> Optional[Dataset]:
        """
        Load dataset from HuggingFace or local file
        
        Args:
            config: Configuration dictionary containing dataset parameters
            split: Dataset split to load
            validation_type: Type of validation split ('matched', 'mismatched', 'both')
            
        Returns:
            Loaded dataset or None if loading fails
        """
        dataset_name = config.get("name", "nyu-mll/multi_nli")
        
        # Handle local HANS dataset
        if dataset_name.lower() in ["hans", "jhu-cogsci/hans"]:
            # Try multiple possible locations for the HANS dataset
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'hans_test_data.csv'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'hans_test_data.csv'),
                'hans_test_data.csv'
            ]
            
            for hans_path in possible_paths:
                if os.path.exists(hans_path):
                    logger.info(f"Loading HANS dataset from local file: {hans_path}")
                    return HANSDataset(hans_path, split=split)
            
            logger.warning(f"HANS dataset file not found in any of the expected locations: {possible_paths}")
            return None
        
        try:
            # Handle MNLI dataset which doesn't have a simple 'validation' split
            if dataset_name.lower() in ["nyu-mll/multi_nli", "multi_nli", "glue/mnli"]:
                if split == "validation":
                    if validation_type == "matched":
                        split = "validation_matched"
                    elif validation_type == "mismatched":
                        split = "validation_mismatched"
                    elif validation_type == "both":
                        # Load both and concatenate
                        matched = load_dataset("nyu-mll/multi_nli", split="validation_matched")
                        mismatched = load_dataset("nyu-mll/multi_nli", split="validation_mismatched")
                        return concatenate_datasets([matched, mismatched])
            
            # Load other datasets
            try:
                dataset = load_dataset(dataset_name, split=split)
                # Ensure the dataset has the required columns
                if not all(field in dataset.column_names for field in ["premise", "hypothesis"]):
                    logger.warning(f"Dataset {dataset_name} is missing required columns (premise, hypothesis)")
                    return None
                return dataset
            except (ValueError, TypeError) as e:
                try:
                    # Fall back to trust_remote_code=True if needed
                    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
                    if not all(field in dataset.column_names for field in ["premise", "hypothesis"]):
                        logger.warning(f"Dataset {dataset_name} is missing required columns (premise, hypothesis)")
                        return None
                    return dataset
                except Exception as inner_e:
                    logger.warning(f"Failed to load dataset {dataset_name} with trust_remote_code=True: {str(inner_e)}")
                    return None
                
        except Exception as e:
            logger.warning(f"Failed to load dataset {dataset_name} (split: {split}): {str(e)}")
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
    
    # def tokenize(self, tokenizer, max_length=512):
    #     """Tokenize dataset"""
    #     def tokenize_function(examples):
    #         return tokenizer(
    #             examples["premise"],
    #             examples["hypothesis"],
    #             truncation=True,
    #             padding="max_length",
    #             max_length=max_length
    #         )
        
    #     tokenized_id = self.id_dataset.map(
    #         tokenize_function,
    #         batched=True
    #     )
        
    #     tokenized_id.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
    #     if self.ood_dataset:
    #         tokenized_ood = self.ood_dataset.map(
    #             tokenize_function,
    #             batched=True
    #         )
    #         tokenized_ood.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    #         return tokenized_id, tokenized_ood
        
    #     return tokenized_id, None
   
    


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
        
        # Handle ID dataset
        if hasattr(self.id_dataset, 'set_format'):
            tokenized_id = self.id_dataset.map(
                tokenize_function,
                batched=True
            )
            tokenized_id.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        else:
            # Handle HANSDataset case
            tokenized_id = {
                "input_ids": [],
                "attention_mask": [],
                "label": []
            }
            for i in range(len(self.id_dataset)):
                item = self.id_dataset[i]
                encoded = tokenizer(
                    item["premise"],
                    item["hypothesis"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt"
                )
                tokenized_id["input_ids"].append(encoded["input_ids"].squeeze(0))
                tokenized_id["attention_mask"].append(encoded["attention_mask"].squeeze(0))
                tokenized_id["label"].append(torch.tensor(item["label"]))
            
            # Convert lists to tensors
            tokenized_id = {k: torch.stack(v) for k, v in tokenized_id.items()}
        
        # Handle OOD dataset if it exists
        tokenized_ood = None
        if self.ood_dataset:
            if hasattr(self.ood_dataset, 'set_format'):
                tokenized_ood = self.ood_dataset.map(
                    tokenize_function,
                    batched=True
                )
                tokenized_ood.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            else:
                # Handle HANSDataset case for OOD
                tokenized_ood = {
                    "input_ids": [],
                    "attention_mask": [],
                    "label": []
                }
                for i in range(len(self.ood_dataset)):
                    item = self.ood_dataset[i]
                    encoded = tokenizer(
                        item["premise"],
                        item["hypothesis"],
                        truncation=True,
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    tokenized_ood["input_ids"].append(encoded["input_ids"].squeeze(0))
                    tokenized_ood["attention_mask"].append(encoded["attention_mask"].squeeze(0))
                    tokenized_ood["label"].append(torch.tensor(item["label"]))
                
                # Convert lists to tensors
                tokenized_ood = {k: torch.stack(v) for k, v in tokenized_ood.items()}
        
        return tokenized_id, tokenized_ood