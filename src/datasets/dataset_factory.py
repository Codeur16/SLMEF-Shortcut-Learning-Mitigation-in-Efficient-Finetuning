"""
Dataset factory for creating different dataset types
"""

import yaml
from typing import Dict, Any, Optional

from .nli_data import NLIDataset
from .sentiment_data import SentimentDataset
from .qa_data import QADataset
from .paraphrase_data import ParaphraseDataset

class DatasetFactory:
    """Factory for creating dataset instances"""
    
    DATASET_CLASSES = {
        "nli": NLIDataset,
        "sentiment": SentimentDataset,
        "qa": QADataset,
        "paraphrase": ParaphraseDataset,
    }
    
    @classmethod
    def create_dataset(
        cls,
        task_type: str,
        config: Dict[str, Any],
        split: str = "validation"
    ):
        """Create a dataset instance"""
        if task_type not in cls.DATASET_CLASSES:
            raise ValueError(f"Unknown task type: {task_type}")
        
        dataset_class = cls.DATASET_CLASSES[task_type]
        return dataset_class(config, split)
    
    @classmethod
    def create_from_config(
        cls,
        task_name: str,
        config_path: str,
        split: str = "validation"
    ):
        """Create dataset from configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls.create_dataset(task_name, config, split)