"""
RG-PEFT trainer with rule-guided regularization
"""

import torch
from .qlora_trainer import QLoRATrainer
from src.rules import RuleEngine
import logging

logger = logging.getLogger(__name__)

class RGPEftTrainer(QLoRATrainer):
    """RG-PEFT trainer with rule-guided regularization"""
    
    def __init__(self, model, config: dict, rule_engine: RuleEngine, device: str = "cuda"):
        super().__init__(model, config, device)
        
        self.rule_engine = rule_engine
        self.lambda_reg = config.get("lambda_reg", 0.1)
        
        logger.info(f"Initialized RG-PEFT trainer with lambda={self.lambda_reg}")
    
    def compute_rule_loss(self, batch, model_outputs):
        """Compute rule-guided regularization loss"""
        if self.rule_engine is None:
            return torch.tensor(0.0, device=self.device)
        
        # Extract texts from batch (implementation depends on task)
        texts = self._extract_texts_from_batch(batch)
        
        # Apply rules
        rule_results = self.rule_engine.apply(texts)
        
        # Compute rule loss
        rule_loss = self.rule_engine.compute_rule_loss(
            model_outputs.logits,
            rule_results,
            self.lambda_reg
        )
        
        return rule_loss
    
    def _extract_texts_from_batch(self, batch):
        """Extract texts from batch for rule application"""
        # This is a placeholder - implementation depends on task format
        # In practice, you would extract the actual text based on task
        return []
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch with rule regularization"""
        self.model.train()
        total_loss = 0
        accumulation_steps = 0
        
        progress_bar = tqdm(dataloader, desc="RG-PEFT Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch_tensors = {k: v.to(self.device) for k, v in batch.items() 
                           if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = self.model(**batch_tensors)
            task_loss = outputs.loss / self.gradient_accumulation_steps
            
            # Compute rule regularization loss
            rule_loss = self.compute_rule_loss(batch, outputs)
            
            # Total loss
            total_batch_loss = task_loss + rule_loss
            
            # Backward pass
            total_batch_loss.backward()
            accumulation_steps += 1
            
            # Update weights if accumulated enough steps
            if accumulation_steps % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += (task_loss.item() + rule_loss.item()) * self.gradient_accumulation_steps
            progress_bar.set_postfix({
                "task_loss": task_loss.item() * self.gradient_accumulation_steps,
                "rule_loss": rule_loss.item() * self.gradient_accumulation_steps
            })
        
        return total_loss / len(dataloader)