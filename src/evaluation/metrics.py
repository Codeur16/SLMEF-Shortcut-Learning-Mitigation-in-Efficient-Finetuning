"""
Evaluation metrics
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def compute_metrics(predictions, labels):
    """Compute classification metrics"""
    if len(predictions) == 0 or len(labels) == 0:
        return {}
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro'),
        "precision_macro": precision_score(labels, predictions, average='macro'),
        "recall_macro": recall_score(labels, predictions, average='macro'),
    }
    
    # For binary classification
    if len(np.unique(labels)) == 2:
        metrics.update({
            "f1_binary": f1_score(labels, predictions, average='binary'),
            "precision_binary": precision_score(labels, predictions, average='binary'),
            "recall_binary": recall_score(labels, predictions, average='binary'),
        })
    
    return metrics

def compute_robustness_gap(id_metrics: dict, ood_metrics: dict) -> float:
    """Compute robustness gap between ID and OOD performance"""
    if "accuracy" not in id_metrics or "accuracy" not in ood_metrics:
        return None
    
    return id_metrics["accuracy"] - ood_metrics["accuracy"]