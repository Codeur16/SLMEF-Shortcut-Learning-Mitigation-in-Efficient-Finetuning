"""
Robustness evaluation
"""

import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def evaluate_robustness(model, id_dataloader, ood_dataloader, device="cuda"):
    """Evaluate model robustness"""
    model.eval()
    
    # Evaluate on ID data
    id_correct = 0
    id_total = 0
    
    with torch.no_grad():
        for batch in tqdm(id_dataloader, desc="ID Evaluation"):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            id_correct += (preds == batch["labels"]).sum().item()
            id_total += len(batch["labels"])
    
    id_accuracy = id_correct / id_total if id_total > 0 else 0
    
    # Evaluate on OOD data
    ood_correct = 0
    ood_total = 0
    
    with torch.no_grad():
        for batch in tqdm(ood_dataloader, desc="OOD Evaluation"):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            ood_correct += (preds == batch["labels"]).sum().item()
            ood_total += len(batch["labels"])
    
    ood_accuracy = ood_correct / ood_total if ood_total > 0 else 0
    
    # Compute robustness metrics
    robustness_gap = id_accuracy - ood_accuracy
    relative_gap = robustness_gap / id_accuracy if id_accuracy > 0 else 0
    
    return {
        "id_accuracy": id_accuracy,
        "ood_accuracy": ood_accuracy,
        "robustness_gap": robustness_gap,
        "relative_gap": relative_gap,
        "id_samples": id_total,
        "ood_samples": ood_total
    }