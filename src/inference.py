"""
Inference script for running predictions on test datasets.
"""

import argparse
import os
import sys
from models import load_model
from dataset import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Run inference on a dataset using a trained model")
    parser.add_argument("--model", required=True, help="Model architecture (flant5, llama, bert, mistral, gpt)")
    parser.add_argument("--task", required=True, help="Task/dataset name (nli, nlu, qa, si)")
    parser.add_argument("--exp_type", required=True, help="Experiment type (efficient-finetuning, neusy-finetuning)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--output_file", required=True, help="Output JSON file path for predictions")
    args = parser.parse_args()

    print(f"Inference parameters: model={args.model}, task={args.task}, exp_type={args.exp_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output file: {args.output_file}")

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")

    # Load model with checkpoint
    model = load_model(args.model, args.exp_type, checkpoint=args.checkpoint)
    
    # Load the model weights (checkpoint is already handled in load())
    print("Loading model from checkpoint...")
    model.load()
    
    # Prepare model for inference (disable training mode)
    if model.model is not None:
        model.model.eval()
        print(" Model loaded and set to evaluation mode")
    
    # Load test dataset (using validation set as test for now)
    print("Loading test dataset...")
    _, test_ds = load_dataset(args.task)
    
    # Run inference
    print("Starting inference...")
    predictions = model.inference(test_ds, args.output_file)
    
    print(f" Inference completed successfully!")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Output saved to: {args.output_file}")


if __name__ == "__main__":
    main()

