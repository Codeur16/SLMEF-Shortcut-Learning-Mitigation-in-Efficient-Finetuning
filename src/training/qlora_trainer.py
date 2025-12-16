"""
QLoRA trainer for efficient fine-tuning
"""

import torch
from .base_trainer import BaseTrainer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class QLoRATrainer(BaseTrainer):
    """QLoRA trainer for parameter-efficient fine-tuning"""
    
    def __init__(self, model, config: dict, device: str = "cuda"):
        super().__init__(model, config, device)
        
        # QLoRA-specific parameters
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        
        # Initialize scheduler
        total_steps = len(self.train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Initialized QLoRA trainer with accumulation steps={self.gradient_accumulation_steps}")
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0
        accumulation_steps = 0
        
        progress_bar = tqdm(dataloader, desc="QLoRA Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            accumulation_steps += 1
            
            # Update weights if accumulated enough steps
            if accumulation_steps % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})
        
        return total_loss / len(dataloader)