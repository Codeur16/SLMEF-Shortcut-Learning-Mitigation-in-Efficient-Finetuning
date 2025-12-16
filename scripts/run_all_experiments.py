#!/usr/bin/env python3
"""
Run all experiments
"""

import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from scripts.run_experiment import run_experiment

logger = setup_logger(__name__)

MODELS = ["bert", "roberta", "flant5", "llama", "mistral"]
TASKS = ["nli", "sentiment", "qa", "paraphrase"]
EXP_TYPES = ["base_eval", "qlora_ft", "rgpeft"]

def run_single_experiment(args):
    """Run single experiment (for parallel execution)"""
    model, task, exp_type, config_dir, output_dir, device, seed = args
    try:
        logger.info(f"Starting: {model}/{task}/{exp_type}")
        results = run_experiment(
            model_name=model,
            task_name=task,
            exp_type=exp_type,
            config_dir=config_dir,
            output_dir=output_dir,
            device=device,
            seed=seed
        )
        logger.info(f"Completed: {model}/{task}/{exp_type}")
        return (model, task, exp_type, "success", results)
    except Exception as e:
        logger.error(f"Failed: {model}/{task}/{exp_type}: {str(e)}")
        return (model, task, exp_type, "failed", str(e))

def run_all_experiments(
    config_dir: str = "configs",
    output_dir: str = "experiments",
    device: str = "cuda",
    seed: int = 42,
    max_workers: int = 2,
    models: list = None,
    tasks: list = None,
    exp_types: list = None
):
    """Run all experiments"""
    
    if models is None:
        models = MODELS
    if tasks is None:
        tasks = TASKS
    if exp_types is None:
        exp_types = EXP_TYPES
    
    # Generate all experiment combinations
    experiments = []
    for model in models:
        for task in tasks:
            for exp_type in exp_types:
                experiments.append((
                    model, task, exp_type,
                    config_dir, output_dir, device, seed
                ))
    
    total_experiments = len(experiments)
    logger.info(f"Running {total_experiments} experiments...")
    logger.info(f"Models: {models}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Experiment types: {exp_types}")
    
    # Run experiments
    results = []
    if max_workers > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_results = executor.map(run_single_experiment, experiments)
            results = list(future_results)
    else:
        # Sequential execution
        for args in experiments:
            results.append(run_single_experiment(args))
    
    # Print summary
    successful = [r for r in results if r[3] == "success"]
    failed = [r for r in results if r[3] == "failed"]
    
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*50)
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if failed:
        logger.info("\nFailed experiments:")
        for f in failed:
            logger.info(f"  {f[0]}/{f[1]}/{f[2]}: {f[4]}")
    
    # Save summary
    import json
    summary = {
        "total_experiments": total_experiments,
        "successful": len(successful),
        "failed": len(failed),
        "failed_details": [
            {
                "model": f[0],
                "task": f[1],
                "exp_type": f[2],
                "error": f[4]
            }
            for f in failed
        ]
    }
    
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run all RG-PEFT experiments")
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_workers", type=int, default=2,
                       help="Number of parallel workers")
    parser.add_argument("--models", type=str, nargs="+", default=MODELS)
    parser.add_argument("--tasks", type=str, nargs="+", default=TASKS)
    parser.add_argument("--exp_types", type=str, nargs="+", default=EXP_TYPES)
    
    args = parser.parse_args()
    
    run_all_experiments(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        max_workers=args.max_workers,
        models=args.models,
        tasks=args.tasks,
        exp_types=args.exp_types
    )

if __name__ == "__main__":
    main()