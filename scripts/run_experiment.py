#!/usr/bin/env python3
"""
Run a single experiment
"""

import argparse
import os
import sys
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_experiment_config
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed
from src.models.factory import ModelFactory
from src.datasets.dataset_factory import DatasetFactory
from src.rules import (
    NLIRuleEngine, SentimentRuleEngine, 
    QARuleEngine, ParaphraseRuleEngine
)
from src.training import QLoRATrainer, RGPEftTrainer
from src.evaluation import Evaluator
# evaluator = Evaluator(model, device)

# Utilisez :
from src.evaluation.evaluator import create_evaluator
logger = setup_logger(__name__)


def determine_dataset_type(task_name: str) -> str:
    """Determine dataset type from task name"""
    task_lower = task_name.lower()
    
    if "hans" in task_lower:
        return "hans"
    elif any(x in task_lower for x in ["nli", "mnli", "snli"]):
        return "nli"
    elif any(x in task_lower for x in ["sentiment", "sst", "imdb"]):
        return "sentiment"
    elif any(x in task_lower for x in ["qa", "question", "squad"]):
        return "qa"
    else:
        return "generic"

def run_experiment(
    model_name="bert",
    task_name="nli",
    exp_type="base_eval",
    config_dir="base_eval",
    output_dir="experiments/bert",
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42
):
    """Run a single experiment"""
    
    # Set seed and device
    set_seed(seed)
    
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load configuration
    logger.info(f"Loading configuration for {model_name}/{task_name}/{exp_type}")
    config = load_experiment_config(model_name, task_name, exp_type, config_dir)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_output_dir = os.path.join(
        output_dir, 
        exp_type,
        model_name,
        task_name,
        timestamp
    )
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Save configuration
    import yaml
    with open(os.path.join(exp_output_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create model
    logger.info(f"Creating {model_name} model...")
    model = ModelFactory.create_model(model_name, config, device)
    
    # Create dataset
    logger.info(f"Creating {task_name} dataset...")
    dataset = DatasetFactory.create_dataset(task_name, config, split="validation")
    
    # Create rule engine (for RG-PEFT)
    rule_engine = None
    if exp_type == "rgpeft":
        logger.info("Creating rule engine...")
        if task_name == "nli":
            rule_engine = NLIRuleEngine()
        elif task_name == "sentiment":
            rule_engine = SentimentRuleEngine()
        elif task_name == "qa":
            rule_engine = QARuleEngine()
        elif task_name == "paraphrase":
            rule_engine = ParaphraseRuleEngine()
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    
    # Déterminer le model_name pour le tokenizer
    model_name_for_tokenizer = config.get("model_name")
    if not model_name_for_tokenizer:
        # Utiliser une valeur par défaut basée sur model_type
        model_type = config.get("model_type", "bert")
        if model_type == "bert":
            model_name_for_tokenizer = "bert-base-uncased"
        elif model_type == "roberta":
            model_name_for_tokenizer = "roberta-base"
        elif model_type == "flan-t5":
            model_name_for_tokenizer = "google/flan-t5-base"
        elif model_type == "llama":
            model_name_for_tokenizer = "meta-llama/Llama-2-7b-hf"
        elif model_type == "mistral":
            model_name_for_tokenizer = "mistralai/Mistral-7B-v0.1"
        else:
            model_name_for_tokenizer = "bert-base-uncased"
        
        logger.warning(f"No model_name specified in config, using default for {model_type}: {model_name_for_tokenizer}")

    # Charger le tokenizer
    tokenizer = ModelFactory.get_tokenizer(config.get("model_type", "bert"), model_name_for_tokenizer)
    
    max_length = config.get("max_length", 512)
    tokenized_id, tokenized_ood = dataset.tokenize(tokenizer, max_length)
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    id_dataloader = DataLoader(
        tokenized_id,
        batch_size=config.get("eval_batch_size", 32),
        shuffle=False
    )
    
    ood_dataloader = None
    if tokenized_ood:
        ood_dataloader = DataLoader(
            tokenized_ood,
            batch_size=config.get("eval_batch_size", 32),
            shuffle=False
        )
    
    # Déterminer le type de dataset
    dataset_type = determine_dataset_type(task_name)
    
    # Run experiment based on type
    if exp_type == "base_eval":
        logger.info("Running base model evaluation...")
        # Créer l'évaluateur approprié
        evaluator = create_evaluator(
            model=model,
            device=device,
            model_type=config.get("model_type", "bert"),
            dataset_type=dataset_type
        )
            
        # Évaluation
        results = evaluator.evaluate_id_ood(id_dataloader, ood_dataloader)

    elif exp_type == "qlora_ft":
        logger.info("Running QLoRA fine-tuning...")
        trainer = QLoRATrainer(model, config, device)
        
        # For simplicity, we'll just evaluate without actual training
        # In practice, you would train here
        evaluator = create_evaluator(
            model=model,
            device=device,
            model_type=config.get("model_type", "bert"),
            dataset_type=dataset_type
        )
        results = evaluator.evaluate_id_ood(id_dataloader, ood_dataloader)
        results["training"] = {"status": "simulated"}
    
    elif exp_type == "rgpeft":
        logger.info("Running RG-PEFT...")
        trainer = RGPEftTrainer(model, config, rule_engine, device)
        
        # Evaluate
        evaluator = create_evaluator(
            model=model,
            device=device,
            model_type=config.get("model_type", "bert"),
            dataset_type=dataset_type
        )
        results = evaluator.evaluate_id_ood(id_dataloader, ood_dataloader)
        results["training"] = {"status": "simulated", "lambda_reg": config.get("lambda_reg", 0.1)}
    
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")
    
    # Save results
    import json
    results_path = os.path.join(exp_output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Experiment completed. Results saved to {results_path}")
    logger.info(f"ID Accuracy: {results.get('id', {}).get('accuracy', 'N/A'):.4f}")
    
    if "ood" in results:
        logger.info(f"OOD Accuracy: {results.get('ood', {}).get('accuracy', 'N/A'):.4f}")
        if "robustness_gap" in results:
            logger.info(f"Robustness Gap: {results['robustness_gap']:.4f}")
    
    return results


# def run_experiment(
#     model_name="bert",
#     task_name="nli",
#     exp_type="base_eval",
#     config_dir="base_eval",
#     output_dir="experiments/bert",
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     seed: int = 42
# ):
#     """Run a single experiment"""
    
#     # Set seed and device
#     set_seed(seed)
    
#     # Auto-detect device if not specified
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.info(f"Using device: {device}")
    
#     # Load configuration
#     logger.info(f"Loading configuration for {model_name}/{task_name}/{exp_type}")
#     config = load_experiment_config(model_name, task_name, exp_type, config_dir)
    
#     # Create output directory
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     exp_output_dir = os.path.join(
#         output_dir, 
#         exp_type,
#         model_name,
#         task_name,
#         timestamp
#     )
#     os.makedirs(exp_output_dir, exist_ok=True)
    
#     # Save configuration
#     import yaml
#     with open(os.path.join(exp_output_dir, "config.yaml"), 'w') as f:
#         yaml.dump(config, f, default_flow_style=False)
    
#     # Create model
#     logger.info(f"Creating {model_name} model...")
#     model = ModelFactory.create_model(model_name, config, device)
    
#     # Create dataset
#     logger.info(f"Creating {task_name} dataset...")
#     dataset = DatasetFactory.create_dataset(task_name, config, split="validation")
    
#     # Create rule engine (for RG-PEFT)
#     rule_engine = None
#     if exp_type == "rgpeft":
#         logger.info("Creating rule engine...")
#         if task_name == "nli":
#             rule_engine = NLIRuleEngine()
#         elif task_name == "sentiment":
#             rule_engine = SentimentRuleEngine()
#         elif task_name == "qa":
#             rule_engine = QARuleEngine()
#         elif task_name == "paraphrase":
#             rule_engine = ParaphraseRuleEngine()
    
#     # Tokenize dataset
#     logger.info("Tokenizing dataset...")
#     tokenizer = ModelFactory.get_tokenizer(model_name, config.get("model_name"))
    





#     # Obtenez le model_name depuis la config, avec une valeur par défaut
#     model_name = config.get("model_name")
#     if not model_name:
#         # Utiliser une valeur par défaut basée sur model_type
#         model_type = config.get("model_type", "bert")
#         if model_type == "bert":
#             model_name = "bert-base-uncased"
#         elif model_type == "roberta":
#             model_name = "roberta-base"
#         elif model_type == "flan-t5":
#             model_name = "google/flan-t5-base"
#         elif model_type == "llama":
#             model_name = "meta-llama/Llama-2-7b-hf"
#         elif model_type == "mistral":
#             model_name = "mistralai/Mistral-7B-v0.1"
#         else:
#             model_name = "bert-base-uncased"
        
#         logger.warning(f"No model_name specified in config, using default for {model_type}: {model_name}")

#     # Maintenant appelez get_tokenizer avec les bons paramètres
#     tokenizer = ModelFactory.get_tokenizer(config.get("model_type", "bert"), model_name)












#     max_length = config.get("max_length", 512)
#     tokenized_id, tokenized_ood = dataset.tokenize(tokenizer, max_length)
    
#     # Create dataloaders
#     from torch.utils.data import DataLoader
    
#     id_dataloader = DataLoader(
#         tokenized_id,
#         batch_size=config.get("eval_batch_size", 32),
#         shuffle=False
#     )
    
#     ood_dataloader = None
#     if tokenized_ood:
#         ood_dataloader = DataLoader(
#             tokenized_ood,
#             batch_size=config.get("eval_batch_size", 32),
#             shuffle=False
#         )
    
#         # Ajoutez cette ligne AVANT le bloc if exp_type == "base_eval":
#     dataset_type = determine_dataset_type(task_name)

#     # Run experiment based on type
#     if exp_type == "base_eval":
#         logger.info("Running base model evaluation...")
#             # Créer l'évaluateur approprié
#         evaluator = create_evaluator(
#             model=model,
#             device=device,
#             model_type=config.get("model_type", "bert"),
#             dataset_type=dataset_type
#         )
            
#        # Évaluation
#         results = evaluator.evaluate_id_ood(id_dataloader, ood_dataloader)

#     elif exp_type == "qlora_ft":
#         logger.info("Running QLoRA fine-tuning...")
#         trainer = QLoRATrainer(model, config, device)
        
#         # For simplicity, we'll just evaluate without actual training
#         # In practice, you would train here
#         evaluator = Evaluator(model, device)
#         results = evaluator.evaluate_id_ood(id_dataloader, ood_dataloader)
#         results["training"] = {"status": "simulated"}
    
#     elif exp_type == "rgpeft":
#         logger.info("Running RG-PEFT...")
#         trainer = RGPEftTrainer(model, config, rule_engine, device)
        
#         # Evaluate
#         evaluator = Evaluator(model, device)
#         results = evaluator.evaluate_id_ood(id_dataloader, ood_dataloader)
#         results["training"] = {"status": "simulated", "lambda_reg": config.get("lambda_reg", 0.1)}
    
#     else:
#         raise ValueError(f"Unknown experiment type: {exp_type}")
    
#     # Save results
#     import json
#     results_path = os.path.join(exp_output_dir, "results.json")
#     with open(results_path, 'w') as f:
#         json.dump(results, f, indent=2)
    
#     logger.info(f"Experiment completed. Results saved to {results_path}")
#     logger.info(f"ID Accuracy: {results.get('id', {}).get('accuracy', 'N/A'):.4f}")
    
#     if "ood" in results:
#         logger.info(f"OOD Accuracy: {results.get('ood', {}).get('accuracy', 'N/A'):.4f}")
#         if "robustness_gap" in results:
#             logger.info(f"Robustness Gap: {results['robustness_gap']:.4f}")
    
#     return results

def main():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--exp_type", type=str, required=True, help="Experiment type")
    parser.add_argument("--config_dir", type=str, default="configs", help="Config directory")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory")
    parser.add_argument("--device", type=str, help="Device to use (cuda or cpu). Auto-detects if not specified.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_experiment(
        model_name=args.model,
        task_name=args.task,
        exp_type=args.exp_type,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )

if __name__ == "__main__":
    main()