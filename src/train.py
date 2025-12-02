# Train code

import argparse
from models import load_model
from dataset import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model architecture to load (flant5, llama, bert, mistral, gpt)." )
    parser.add_argument("--task", required=True, help="Downstream task/dataset to load (nli, nlu, qa, si).")
    parser.add_argument("--exp_type", required=True, help="Experiment type (efficient-finetuning, neusy-finetuning).")
    parser.add_argument("--save_dir", required=True, help="Directory where checkpoints will be saved.")
    args = parser.parse_args()

    print(f"Parameters: model={args.model}, task={args.task}, exp_type={args.exp_type}, save_dir={args.save_dir}")

    # Load model instance
    model = load_model(args.model, args.exp_type)

    # Load dataset (returns train, val)
    train_ds, val_ds = load_dataset(args.task)

    # Prepare and run training using the model's API
    model.load()
    model.prepare_for_training()
    model.train(train_ds, val_ds, args.save_dir)


if __name__ == "__main__":
    main()