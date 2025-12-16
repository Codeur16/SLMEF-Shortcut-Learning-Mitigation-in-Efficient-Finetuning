"""
Metrics computation for evaluation
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def compute_metrics(predictions: List, labels: List, dataset_type: str = "nli") -> Dict:
    """Compute metrics based on dataset type"""
    if not predictions or not labels:
        logger.warning("Empty predictions or labels provided")
        return {}
    
    # Convert to numpy arrays
    preds_np = np.array(predictions)
    labels_np = np.array(labels)
    
    # Basic metrics
    acc = accuracy_score(labels_np, preds_np)
    
    # Determine number of classes based on dataset type
    if dataset_type.lower() == "hans":
        # HANS is binary classification
        num_classes = 2
        labels_list = [0, 1]
        label_names = ['entailment', 'non-entailment']
    elif dataset_type.lower() in ["nli", "mnli"]:
        # MNLI is 3-class classification
        num_classes = 3
        labels_list = [0, 1, 2]
        label_names = ['entailment', 'neutral', 'contradiction']
    else:
        # Generic case
        num_classes = len(np.unique(labels_np))
        labels_list = list(range(num_classes))
        label_names = [f'class_{i}' for i in range(num_classes)]
    
    try:
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
        
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return {"accuracy": float(acc)}
    
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
    
    # Add dataset-specific metrics
    if dataset_type.lower() in ["nli", "mnli", "hans"]:
        metrics.update(compute_nli_specific_metrics(preds_np, labels_np, dataset_type))
    
    return metrics

def compute_nli_specific_metrics(predictions: np.ndarray, labels: np.ndarray, 
                                dataset_type: str = "nli") -> Dict:
    """Compute NLI-specific metrics"""
    metrics = {}
    
    if dataset_type.lower() == "hans":
        # HANS-specific metrics
        # Accuracy on different heuristics would require additional data
        metrics["hans_accuracy"] = float(np.mean(predictions == labels))
        
    elif dataset_type.lower() in ["nli", "mnli"]:
        # MNLI-specific metrics
        # Accuracy per relation type
        relations = {
            'entailment': 0,
            'neutral': 1, 
            'contradiction': 2
        }
        
        for relation, idx in relations.items():
            mask = labels == idx
            if np.sum(mask) > 0:
                rel_acc = np.mean(predictions[mask] == labels[mask])
                metrics[f"accuracy_{relation}"] = float(rel_acc)
    
    return metrics

def compute_robustness_gap(id_metrics: Dict, ood_metrics: Dict) -> Dict:
    """Compute robustness gap between ID and OOD metrics"""
    if not id_metrics or not ood_metrics:
        return {}
    
    gap_metrics = {}
    
    # Accuracy gap
    if "accuracy" in id_metrics and "accuracy" in ood_metrics:
        gap_metrics["accuracy_gap"] = id_metrics["accuracy"] - ood_metrics["accuracy"]
        
        # Relative performance drop
        if id_metrics["accuracy"] > 0:
            gap_metrics["relative_drop"] = (
                gap_metrics["accuracy_gap"] / id_metrics["accuracy"]
            ) * 100
    
    # F1 gap
    if "f1_weighted" in id_metrics and "f1_weighted" in ood_metrics:
        gap_metrics["f1_gap"] = id_metrics["f1_weighted"] - ood_metrics["f1_weighted"]
    
    # Per-class gaps if available
    if "per_class" in id_metrics and "per_class" in ood_metrics:
        class_gaps = {}
        for class_name in id_metrics["per_class"]:
            if class_name in ood_metrics["per_class"]:
                id_f1 = id_metrics["per_class"][class_name].get("f1", 0)
                ood_f1 = ood_metrics["per_class"][class_name].get("f1", 0)
                class_gaps[class_name] = id_f1 - ood_f1
        
        if class_gaps:
            gap_metrics["per_class_gaps"] = class_gaps
    
    return gap_metrics
    
        
            
                # """
# Evaluation metrics
# """

# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# import numpy as np

# def compute_metrics(predictions, labels):
#     """Compute classification metrics"""
#     if len(predictions) == 0 or len(labels) == 0:
#         return {}
    
#     predictions = np.array(predictions)
#     labels = np.array(labels)
    
#     metrics = {
#         "accuracy": accuracy_score(labels, predictions),
#         "f1_macro": f1_score(labels, predictions, average='macro'),
#         "precision_macro": precision_score(labels, predictions, average='macro'),
#         "recall_macro": recall_score(labels, predictions, average='macro'),
#     }
    
#     # For binary classification
#     if len(np.unique(labels)) == 2:
#         metrics.update({
#             "f1_binary": f1_score(labels, predictions, average='binary'),
#             "precision_binary": precision_score(labels, predictions, average='binary'),
#             "recall_binary": recall_score(labels, predictions, average='binary'),
#         })
    
#     return metrics

# def compute_robustness_gap(id_metrics: dict, ood_metrics: dict) -> float:
#     """Compute robustness gap between ID and OOD performance"""
#     if "accuracy" not in id_metrics or "accuracy" not in ood_metrics:
#         return None
    
#     return id_metrics["accuracy"] - ood_metrics["accuracy"]