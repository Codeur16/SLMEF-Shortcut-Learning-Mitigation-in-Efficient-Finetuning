"""
HANS (Heuristic Analysis for NLI Systems) dataset from local CSV file
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class HANSDataset(Dataset):
    """HANS dataset loaded from local CSV file"""
    
    def __init__(self, file_path: str, split: str = "validation"):
        """
        Args:
            file_path: Path to the HANS CSV file
            split: Split to load (not used for HANS as it's a test set)
        """
        self.file_path = file_path
        self.data = self._load_data()
        
    def _load_data(self):
        """Load and preprocess the HANS dataset from CSV"""
        try:
            # Load the CSV file
            df = pd.read_csv(self.file_path)
            
            # Convert to list of dicts for consistency with other datasets
            data = []
            for _, row in df.iterrows():
                data.append({
                    'premise': str(row.get('premise', '')),
                    'hypothesis': str(row.get('hypothesis', '')),
                    'label': self._map_label(row.get('label', -1)),
                    'pair_id': row.get('pairID', '')
                })
                
            logger.info(f"Loaded {len(data)} examples from {self.file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading HANS dataset from {self.file_path}: {str(e)}")
            return []
    
    def _map_label(self, label):
        """Map HANS labels to standard NLI labels (0: entailment, 1: neutral, 2: contradiction)"""
        # HANS uses 'entailment' and 'non-entailment', map to standard NLI labels
        if label == 'entailment':
            return 0  # entailment
        return 1  # neutral (treat non-entailment as neutral for compatibility)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
