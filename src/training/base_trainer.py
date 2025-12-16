"""
Base trainer class
"""

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import os

logger = logging.getLogger(__name__)

class BaseTrainer:
    """Base trainer for model training"""
    
    def __init__(self, model, config: dict, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = device
        
        # Training parameters
        self.learning_rate = config.get("learning_rate", 2e-4)
        self.batch_size = config.get("batch_size", 32)
        self.num_epochs = config.get("num_epochs", 10)
        self.warmup_ratio = config.get("warmup_ratio", 0.06)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        logger.info(f"Initialized trainer with lr={self.learning_rate}, batch_size={self.batch_size}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        from src.evaluation.metrics import compute_metrics
        metrics = compute_metrics(all_preds, all_labels)
        metrics["loss"] = total_loss / len(dataloader)
        
        return metrics
    
    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics
        }
        
        torch.save(checkpoint, os.path.join(path, f"checkpoint_epoch_{epoch}.pt"))
        logger.info(f"Saved checkpoint to {path}")