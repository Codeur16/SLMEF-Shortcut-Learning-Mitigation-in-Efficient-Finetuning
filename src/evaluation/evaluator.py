"""
Model evaluator with support for different dataset types
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json
from typing import Dict, List, Optional, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)

class Evaluator:
    """Model evaluator with support for different dataset types"""
    
    def __init__(self, model, device: str = "cuda", model_type: str = "bert", dataset_type: str = "nli"):
        self.model = model
        self.device = device
        self.model_type = model_type  # "bert", "roberta", "flan-t5", etc.
        self.dataset_type = dataset_type  # "nli", "hans", "mnli"
        
    def _prepare_batch_for_model(self, batch: Dict) -> Dict:
        """Prepare batch for different model types"""
        if self.model_type in ["bert", "roberta", "llama", "mistral"]:
            # Encoder models expect input_ids, attention_mask
            model_inputs = {
                'input_ids': batch.get('input_ids'),
                'attention_mask': batch.get('attention_mask')
            }
            
            # Add token_type_ids for BERT if available
            if 'token_type_ids' in batch:
                model_inputs['token_type_ids'] = batch['token_type_ids']
                
        elif self.model_type in ["flan-t5"]:
            # T5 models expect input_ids, attention_mask for encoder
            # and labels for decoder during training
            model_inputs = {
                'input_ids': batch.get('input_ids'),
                'attention_mask': batch.get('attention_mask')
            }
            
            # For evaluation, we might not need labels
            if 'labels' in batch:
                model_inputs['labels'] = batch['labels']
        else:
            # Generic case
            model_inputs = {k: v for k, v in batch.items() 
                          if k not in ['label', 'labels', 'premise', 'hypothesis', 'text']}
        
        return model_inputs
    
    def _get_labels_from_batch(self, batch: Dict) -> Optional[torch.Tensor]:
        """Extract labels from batch based on dataset type"""
        if 'label' in batch:
            return batch['label']
        elif 'labels' in batch:
            return batch['labels']
        return None
    
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate model on dataset"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch_count += 1
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Extract labels before forward pass
                labels = self._get_labels_from_batch(batch)
                
                # Prepare inputs for model (remove label/labels from forward call)
                model_inputs = self._prepare_batch_for_model(batch)
                
                try:
                    # Forward pass
                    outputs = self.model(**model_inputs)
                    
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        total_loss += outputs.loss.item()
                    
                    # Get predictions based on model type
                    if hasattr(outputs, 'logits'):
                        preds = torch.argmax(outputs.logits, dim=-1)
                        all_preds.extend(preds.cpu().numpy())
                    elif hasattr(outputs, 'predictions'):
                        # Some models might use different attribute names
                        preds = torch.argmax(outputs.predictions, dim=-1)
                        all_preds.extend(preds.cpu().numpy())
                    
                    if labels is not None:
                        all_labels.extend(labels.cpu().numpy())
                        
                except Exception as e:
                    logger.warning(f"Error in batch {batch_count}: {e}")
                    continue
        
        # Compute metrics
        metrics = self.compute_metrics(all_preds, all_labels)
        
        if batch_count > 0:
            metrics["loss"] = total_loss / batch_count
        
        return metrics
    
    def compute_metrics(self, predictions: List, labels: List) -> Dict:
        """Compute metrics based on dataset type"""
        if not predictions or not labels:
            return {}
        
        # Convert to numpy arrays
        preds_np = np.array(predictions)
        labels_np = np.array(labels)
        
        # Basic metrics
        acc = accuracy_score(labels_np, preds_np)
        
        # Determine number of classes based on dataset type
        if self.dataset_type == "hans":
            # HANS is binary classification
            num_classes = 2
            labels_list = [0, 1]
            label_names = ['entailment', 'non-entailment']
        elif self.dataset_type in ["nli", "mnli"]:
            # MNLI is 3-class classification
            num_classes = 3
            labels_list = [0, 1, 2]
            label_names = ['entailment', 'neutral', 'contradiction']
        else:
            # Default binary
            num_classes = len(np.unique(labels_np))
            labels_list = list(range(num_classes))
            label_names = [f'class_{i}' for i in range(num_classes)]
        
        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_np, preds_np, labels=labels_list, average=None
        )
        
        # Weighted average F1
        f1_weighted = precision_recall_fscore_support(
            labels_np, preds_np, average='weighted'
        )[2]
        
        # Macro average F1
        f1_macro = precision_recall_fscore_support(
            labels_np, preds_np, average='macro'
        )[2]
        
        # Confusion matrix
        cm = confusion_matrix(labels_np, preds_np, labels=labels_list)
        
        # Format results
        metrics = {
            "accuracy": float(acc),
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_macro),
            "confusion_matrix": cm.tolist(),
            "per_class": {
                label_names[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i])
                }
                for i in range(num_classes)
            }
        }
        
        # Add semantic metrics for NLI tasks
        if self.dataset_type in ["nli", "mnli", "hans"]:
            metrics.update(self.compute_semantic_metrics())
        
        return metrics
    
    def compute_semantic_metrics(self) -> Dict:
        """Compute semantic metrics for NLI tasks (optional)"""
        # These would require access to the original text data
        # For now, return empty dict
        return {
            "semantic_fidelity_score": 0.0,
            "label_consistency_score": 0.0,
            "composite_score": 0.0
        }
    
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
            
            # Compute performance drop percentage
            if "accuracy" in id_metrics and id_metrics["accuracy"] > 0:
                performance_drop = (robustness_gap / id_metrics["accuracy"]) * 100
                results["performance_drop_percent"] = performance_drop
                logger.info(f"Performance drop: {performance_drop:.2f}%")
        
        return results
    
    def evaluate_multiple_models(self, dataloader: DataLoader, models_dict: Dict[str, torch.nn.Module]) -> Dict:
        """Evaluate multiple models on the same dataset"""
        results = {}
        
        for model_name, model in models_dict.items():
            logger.info(f"Evaluating model: {model_name}")
            self.model = model.to(self.device)
            metrics = self.evaluate(dataloader)
            results[model_name] = metrics
        
        return results
    
    def evaluate_with_thresholds(self, dataloader: DataLoader, confidence_thresholds: List[float] = None) -> Dict:
        """Evaluate with different confidence thresholds"""
        if confidence_thresholds is None:
            confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = {}
        
        for threshold in confidence_thresholds:
            logger.info(f"Evaluating with confidence threshold: {threshold}")
            # Implementation would depend on getting confidence scores from model
            # Placeholder for now
            results[f"threshold_{threshold}"] = {"accuracy": 0.0}
        
        return results
    
    def save_results(self, results: dict, path: str):
        """Save evaluation results"""
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {path}")


class NLIEvaluator(Evaluator):
    """Specialized evaluator for NLI tasks"""
    
    def __init__(self, model, device: str = "cuda", model_type: str = "bert"):
        super().__init__(model, device, model_type, dataset_type="nli")
    
    def compute_semantic_metrics(self, premises: List[str] = None, 
                                 hypotheses: List[str] = None,
                                 predictions: List[int] = None) -> Dict:
        """Compute advanced semantic metrics for NLI"""
        if premises is None or hypotheses is None or predictions is None:
            return super().compute_semantic_metrics()
        
        try:
            # Optional: Add semantic similarity and consistency metrics
            # These require additional imports
            from sentence_transformers import SentenceTransformer, util
            
            # Compute semantic similarity between premise and hypothesis
            model = SentenceTransformer("all-MiniLM-L6-v2")
            inputs = [f"{p} {h}" for p, h in zip(premises, hypotheses)]
            emb_inputs = model.encode(inputs, convert_to_tensor=True)
            
            # For simplicity, return placeholder metrics
            semantic_score = float(torch.mean(util.cos_sim(emb_inputs, emb_inputs)).item())
            
            return {
                "semantic_fidelity_score": semantic_score,
                "label_consistency_score": 0.0,  # Would require RoBERTa-MNLI model
                "composite_score": 0.5 * semantic_score
            }
            
        except ImportError:
            logger.warning("SentenceTransformers not installed, skipping semantic metrics")
            return super().compute_semantic_metrics()


class HANSEvaluator(Evaluator):
    """Specialized evaluator for HANS dataset"""
    
    def __init__(self, model, device: str = "cuda", model_type: str = "bert"):
        super().__init__(model, device, model_type, dataset_type="hans")
    
    def compute_heuristic_metrics(self, df_with_heuristics: 'pd.DataFrame' = None) -> Dict:
        """Compute metrics specific to HANS heuristics"""
        if df_with_heuristics is None:
            return {}
        
        # HANS-specific metrics: accuracy per heuristic type
        heuristics = df_with_heuristics['heuristic'].unique()
        metrics = {}
        
        for heuristic in heuristics:
            subset = df_with_heuristics[df_with_heuristics['heuristic'] == heuristic]
            if len(subset) > 0:
                correct = subset[subset['predicted'] == subset['label']]
                acc = len(correct) / len(subset)
                metrics[f"accuracy_{heuristic}"] = float(acc)
        
        return metrics


def create_evaluator(model, device: str = "cuda", model_type: str = "bert", 
                     dataset_type: str = "nli") -> Evaluator:
    """Factory function to create appropriate evaluator"""
    if dataset_type.lower() == "hans":
        return HANSEvaluator(model, device, model_type)
    elif dataset_type.lower() in ["nli", "mnli"]:
        return NLIEvaluator(model, device, model_type)
    else:
        return Evaluator(model, device, model_type, dataset_type)
    
        
            
            
            
            
            
            
            
            
            
            # """
# Model evaluator
# """

# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import logging
# import json

# logger = logging.getLogger(__name__)

# class Evaluator:
#     """Model evaluator"""
    
#     def __init__(self, model, device: str = "cuda"):
#         self.model = model
#         self.device = device
    
#     def evaluate(self, dataloader: DataLoader) -> dict:
#         """Evaluate model on dataset"""
#         self.model.eval()
#         all_preds = []
#         all_labels = []
#         total_loss = 0
        
#         with torch.no_grad():
#             for batch in tqdm(dataloader, desc="Evaluating"):
#                 # Move batch to device
#                 batch = {k: v.to(self.device) for k, v in batch.items() 
#                         if isinstance(v, torch.Tensor)}
                
#                 # Forward pass
#                 outputs = self.model(**batch)
                
#                 if hasattr(outputs, 'loss') and outputs.loss is not None:
#                     total_loss += outputs.loss.item()
                
#                 # Get predictions
#                 if hasattr(outputs, 'logits'):
#                     preds = torch.argmax(outputs.logits, dim=-1)
#                     all_preds.extend(preds.cpu().numpy())
                
#                 if "labels" in batch:
#                     all_labels.extend(batch["labels"].cpu().numpy())
        
#         # Compute metrics
#         from .metrics import compute_metrics
#         metrics = compute_metrics(all_preds, all_labels)
        
#         if len(dataloader) > 0:
#             metrics["loss"] = total_loss / len(dataloader)
        
#         return metrics
    
#     def evaluate_id_ood(self, id_dataloader: DataLoader, ood_dataloader: DataLoader = None) -> dict:
#         """Evaluate on in-distribution and out-of-distribution data"""
#         results = {}
        
#         # Evaluate ID
#         logger.info("Evaluating on in-distribution data...")
#         id_metrics = self.evaluate(id_dataloader)
#         results["id"] = id_metrics
        
#         # Evaluate OOD if available
#         if ood_dataloader:
#             logger.info("Evaluating on out-of-distribution data...")
#             ood_metrics = self.evaluate(ood_dataloader)
#             results["ood"] = ood_metrics
            
#             # Compute robustness gap
#             if "accuracy" in id_metrics and "accuracy" in ood_metrics:
#                 robustness_gap = id_metrics["accuracy"] - ood_metrics["accuracy"]
#                 results["robustness_gap"] = robustness_gap
#                 logger.info(f"Robustness gap: {robustness_gap:.4f}")
        
#         return results
    
#     def save_results(self, results: dict, path: str):
#         """Save evaluation results"""
#         with open(path, 'w') as f:
#             json.dump(results, f, indent=2)
#         logger.info(f"Saved results to {path}")