# Train code

import argparse
from models.flant5 import FlanT5Model

def load_dataset_for_task(task):
    """Load the appropriate dataset based on the task."""
    try:
        if task == 'nli':
            from dataset.nli_dataset import load_nli_dataset as loader
        elif task == 'nlu':
            from dataset.nlu import load_nlu_dataset as loader
        elif task == 'qa':
            from dataset.qa import load_qa_dataset as loader
        elif task == 'si':
            from dataset.si import load_si_dataset as loader
        else:
            raise ValueError(f"Unsupported task: {task}")
        return loader()
    except ImportError as e:
        raise ImportError(f"Failed to import dataset loader for task '{task}': {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model architecture to load (flant5, llama, bert, mistral, gpt).")
    parser.add_argument("--task", required=True, help="Downstream task/dataset to load (nli, nlu, qa, si).")
    parser.add_argument("--exp_type", required=True, help="Experiment type (efficient-finetuning, neusy-finetuning).")
    parser.add_argument("--save_dir", required=True, help="Directory where checkpoints will be saved.")
    args = parser.parse_args()

    print(f"Parameters: model={args.model}, task={args.task}, exp_type={args.exp_type}, save_dir={args.save_dir}")

    # Initialize model
    model = FlanT5Model(model_name=args.model, exp_type=args.exp_type)

    # Load dataset (returns train, val)
    try:
        train_ds, val_ds = load_dataset_for_task(args.task)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset for task '{args.task}': {str(e)}")

    # Prepare and run training using the model's API
    model.load()
    model.prepare_for_training()
    model.train(train_ds, val_ds, args.save_dir)


if __name__ == "__main__":
    main()