"""
Model evaluator
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json

logger = logging.getLogger(__name__)

class Evaluator:
    """Model evaluator"""
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
    
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate model on dataset"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() 
                        if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(**batch)
                
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    total_loss += outputs.loss.item()
                
                # Get predictions
                if hasattr(outputs, 'logits'):
                    preds = torch.argmax(outputs.logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                
                if "labels" in batch:
                    all_labels.extend(batch["labels"].cpu().numpy())
        
        # Compute metrics
        from .metrics import compute_metrics
        metrics = compute_metrics(all_preds, all_labels)
        
        if len(dataloader) > 0:
            metrics["loss"] = total_loss / len(dataloader)
        
        return metrics
    
    def evaluate_id_ood(self, id_dataloader: DataLoader, ood_dataloader: DataLoader = None) -> dict:
        """Evaluate on in-distribution and out-of-distribution data"""
        results = {}
        
        # Evaluate ID
        logger.info("Evaluating on in-distribution data...")
        id_metrics = self.evaluate(id_dataloader)
        results["id"] = id_metrics
        
        # Evaluate OOD if available
        if ood_dataloader:
            logger.info("Evaluating on out-of-distribution data...")
            ood_metrics = self.evaluate(ood_dataloader)
            results["ood"] = ood_metrics
            
            # Compute robustness gap
            if "accuracy" in id_metrics and "accuracy" in ood_metrics:
                robustness_gap = id_metrics["accuracy"] - ood_metrics["accuracy"]
                results["robustness_gap"] = robustness_gap
                logger.info(f"Robustness gap: {robustness_gap:.4f}")
        
        return results
    
    def save_results(self, results: dict, path: str):
        """Save evaluation results"""
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {path}")