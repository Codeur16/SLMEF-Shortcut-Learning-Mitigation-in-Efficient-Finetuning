"""
Evaluation script for computing metrics (accuracy, F1-score) on predictions.
"""

import argparse
import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_predictions(predictions_file):
    """Load predictions from JSON file."""
    with open(predictions_file, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(predictions):
    """Compute accuracy and F1-score from predictions."""
    y_true = []
    y_pred = []
    
    for pred in predictions:
        true_label = pred.get("true_label")
        pred_label = pred.get("predicted_label")
        
        # Only include predictions where we have both true and predicted labels
        if true_label is not None and pred_label is not None:
            y_true.append(int(true_label))
            y_pred.append(int(pred_label))
    
    if len(y_true) == 0:
        raise ValueError("No valid predictions found. Check prediction format.")
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    
    # Per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": [float(f) for f in f1_per_class] if f1_per_class is not None else None,
        "classification_report": class_report,
        "n_samples": len(y_true),
        "n_valid_predictions": len(y_true),
        "n_total_predictions": len(predictions)
    }


def save_metrics(metrics, metrics_file):
    """Save metrics to CSV file."""
    os.makedirs(os.path.dirname(metrics_file) if os.path.dirname(metrics_file) else ".", exist_ok=True)
    
    # Prepare data for CSV
    metrics_data = {
        "metric": [
            "accuracy",
            "f1_macro",
            "f1_micro",
            "f1_weighted",
            "n_samples",
            "n_valid_predictions",
            "n_total_predictions"
        ],
        "value": [
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics["f1_micro"],
            metrics["f1_weighted"],
            metrics["n_samples"],
            metrics["n_valid_predictions"],
            metrics["n_total_predictions"]
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    df.to_csv(metrics_file, index=False)
    
    # Also save detailed report as JSON
    json_file = metrics_file.replace(".csv", "_detailed.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Metrics saved to {metrics_file}")
    print(f"✅ Detailed report saved to {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions and compute metrics")
    parser.add_argument("--task", required=True, help="Task name (nli, nlu, qa, si)")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON file")
    parser.add_argument("--metrics_file", required=True, help="Path to output metrics CSV file")
    args = parser.parse_args()

    print(f"Evaluation parameters: task={args.task}")
    print(f"Predictions file: {args.predictions}")
    print(f"Metrics file: {args.metrics_file}")

    # Check if predictions file exists
    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")

    # Load predictions
    print("Loading predictions...")
    predictions = load_predictions(args.predictions)
    print(f"   Loaded {len(predictions)} predictions")

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(predictions)

    # Display results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Micro): {metrics['f1_micro']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    if metrics['f1_per_class']:
        print(f"\nF1-Score per class: {metrics['f1_per_class']}")
    
    print(f"\nSamples: {metrics['n_samples']} valid / {metrics['n_total_predictions']} total")
    print("="*50)

    # Save metrics
    save_metrics(metrics, args.metrics_file)

    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()

