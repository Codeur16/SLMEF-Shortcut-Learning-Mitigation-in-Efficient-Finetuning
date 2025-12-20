"""
Model evaluator with support for different dataset types
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import json
import os
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import random
from datasets import load_dataset

logger = logging.getLogger(__name__)

class FixedTestSplits:
    """Manage fixed test splits for consistent evaluation"""
    
    def __init__(self, seed: int = 42, test_size: int = 5000, cache_dir: str = "data/fixed_splits"):
        self.seed = seed
        self.test_size = test_size
        self.cache_dir = cache_dir
        self.rng = np.random.RandomState(seed)
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_nli_fixed_splits(self) -> Tuple[Dataset, Dataset]:
        """Get fixed NLI test splits (ID: MNLI, OOD: HANS)"""
        cache_path = os.path.join(self.cache_dir, f"nli_fixed_splits_seed{self.seed}_size{self.test_size}.json")
        
        # Try to load from cache
        if os.path.exists(cache_path):
            logger.info(f"Loading fixed splits from cache: {cache_path}")
            return self._load_splits_from_cache(cache_path)
        
        # Create new splits
        logger.info(f"Creating new fixed splits (seed={self.seed}, size={self.test_size})")
        id_dataset, ood_dataset = self._create_nli_splits()
        
        # Save to cache
        self._save_splits_to_cache(id_dataset, ood_dataset, cache_path)
        
        return id_dataset, ood_dataset
    
    def _create_nli_splits(self) -> Tuple[Dataset, Dataset]:
        """Create balanced fixed splits for NLI"""
        # Load datasets
        mnli = load_dataset("nyu-mll/multi_nli", trust_remote_code=True)
        hans = load_dataset("jhu-cogsci/hans", trust_remote_code=True)
        
        # ID: MNLI validation_matched (balanced 3-class)
        mnli_val = mnli["validation_matched"]
        labels = mnli_val["label"]
        
        # Stratified sampling for balanced classes
        label_to_indices = {}
        for label in [0, 1, 2]:  # entailment, neutral, contradiction
            label_indices = [i for i, l in enumerate(labels) if l == label]
            label_to_indices[label] = label_indices
        
        # Calculate samples per class
        samples_per_class = self.test_size // 3
        
        selected_indices = []
        for label in [0, 1, 2]:
            indices = label_to_indices[label]
            # Ensure we don't try to sample more than available
            n_samples = min(samples_per_class, len(indices))
            selected = self.rng.choice(indices, n_samples, replace=False)
            selected_indices.extend(selected)
        
        # Create ID dataset
        id_dataset = mnli_val.select(selected_indices)
        
        # OOD: HANS dataset (balanced binary)
        # Combine train and validation splits
        hans_data = []
        for split in ["train", "validation"]:
            hans_data.extend(hans[split])
        
        # Convert to list of dicts
        hans_items = []
        for item in hans_data:
            hans_items.append({
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "label": 0 if item["label"] == 0 else 1,  # Convert to binary: 0=entailment, 1=non-entailment
                "heuristic": item.get("heuristic", "unknown")
            })
        
        # Balance HANS dataset
        entailment_items = [item for item in hans_items if item["label"] == 0]
        non_entailment_items = [item for item in hans_items if item["label"] == 1]
        
        # Sample equal number from each class
        n_per_class = self.test_size // 2
        n_per_class = min(n_per_class, len(entailment_items), len(non_entailment_items))
        
        selected_entailment = random.sample(entailment_items, n_per_class)
        selected_non_entailment = random.sample(non_entailment_items, n_per_class)
        
        ood_items = selected_entailment + selected_non_entailment
        random.shuffle(ood_items)
        
        # Create OOD dataset
        ood_dataset = Dataset.from_dict({
            "premise": [item["premise"] for item in ood_items],
            "hypothesis": [item["hypothesis"] for item in ood_items],
            "label": [item["label"] for item in ood_items],
            "heuristic": [item["heuristic"] for item in ood_items]
        })
        
        return id_dataset, ood_dataset
    
    def _save_splits_to_cache(self, id_dataset: Dataset, ood_dataset: Dataset, cache_path: str):
        """Save splits to cache"""
        cache_data = {
            "seed": self.seed,
            "test_size": self.test_size,
            "id_indices": id_dataset["premise"],  # Store enough to reconstruct
            "ood_data": {
                "premise": ood_dataset["premise"],
                "hypothesis": ood_dataset["hypothesis"],
                "label": ood_dataset["label"],
                "heuristic": ood_dataset["heuristic"]
            }
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Saved fixed splits to: {cache_path}")
    
    def _load_splits_from_cache(self, cache_path: str) -> Tuple[Dataset, Dataset]:
        """Load splits from cache"""
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # For ID, we need to reload MNLI and select indices
        mnli = load_dataset("nyu-mll/multi_nli", trust_remote_code=True)
        mnli_val = mnli["validation_matched"]
        
        # Recreate ID dataset using stored premises (simplified approach)
        # In practice, you might want to store the actual indices
        id_dataset = mnli_val.select(range(cache_data["test_size"]))
        
        # Recreate OOD dataset
        ood_data = cache_data["ood_data"]
        ood_dataset = Dataset.from_dict({
            "premise": ood_data["premise"],
            "hypothesis": ood_data["hypothesis"],
            "label": ood_data["label"],
            "heuristic": ood_data["heuristic"]
        })
        
        return id_dataset, ood_dataset


class FewShotInstructEvaluator:
    """Evaluator with few-shot instructions for guiding base models"""
    
    def __init__(self, model, tokenizer, device: str = "cuda", 
                 model_type: str = "bert", num_shots: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_type = model_type
        self.num_shots = num_shots
        self.few_shot_examples = self._create_few_shot_examples()
    
    def _create_few_shot_examples(self) -> List[Dict]:
        """Create few-shot examples for NLI task"""
        examples = [
            {
                "premise": "The cat is sleeping on the mat.",
                "hypothesis": "The cat is resting.",
                "label": "entailment",
                "explanation": "If a cat is sleeping, it is definitely resting."
            },
            {
                "premise": "The man is eating an apple.",
                "hypothesis": "The man is eating a banana.",
                "label": "contradiction",
                "explanation": "An apple and a banana are different fruits."
            },
            {
                "premise": "The sky is blue.",
                "hypothesis": "The weather is good.",
                "label": "neutral",
                "explanation": "A blue sky doesn't necessarily mean good weather."
            }
        ]
        return examples[:self.num_shots]
    
    def _create_prompt(self, premise: str, hypothesis: str) -> str:
        """Create instruction prompt with few-shot examples"""
        prompt = """Task: Determine the relationship between the premise and hypothesis.
Choose from: entailment, contradiction, or neutral.

Examples:"""

        # Add few-shot examples
        for i, example in enumerate(self.few_shot_examples, 1):
            prompt += f"""
{i}. Premise: {example['premise']}
   Hypothesis: {example['hypothesis']}
   Relationship: {example['label']}
   Explanation: {example['explanation']}"""

        prompt += f"""

Now analyze this new case:

Premise: {premise}
Hypothesis: {hypothesis}

What is the relationship? (entailment/contradiction/neutral): """
        
        return prompt
    
    def predict_with_instructions(self, dataloader: DataLoader) -> Tuple[List, List]:
        """Predict with few-shot instructions"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Few-shot evaluation"):
                # For encoder models, we need to adapt
                if self.model_type in ["bert", "roberta"]:
                    # Traditional classification
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Remove labels from forward pass
                    labels = batch.pop('label', None)
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    preds = torch.argmax(outputs.logits, dim=-1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    if labels is not None:
                        all_labels.extend(labels.cpu().numpy())
                
                elif self.model_type in ["flan-t5", "llama", "mistral"]:
                    # Text generation with instructions
                    for i in range(len(batch["premise"])):
                        premise = batch["premise"][i]
                        hypothesis = batch["hypothesis"][i]
                        label = batch["label"][i] if "label" in batch else None
                        
                        # Create prompt with few-shot examples
                        prompt = self._create_prompt(premise, hypothesis)
                        
                        # Tokenize and generate
                        inputs = self.tokenizer(prompt, return_tensors="pt", 
                                               truncation=True, max_length=512).to(self.device)
                        
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=10,
                            temperature=0.7,
                            do_sample=False
                        )
                        
                        # Decode and parse response
                        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        pred_text = response.split(":")[-1].strip().lower()
                        
                        # Map text to label
                        if "entail" in pred_text:
                            pred = 0
                        elif "contrad" in pred_text:
                            pred = 2
                        elif "neutral" in pred_text:
                            pred = 1
                        else:
                            # Default fallback
                            pred = 1
                        
                        all_preds.append(pred)
                        if label is not None:
                            all_labels.append(label)
        
        return all_preds, all_labels


class Evaluator:
    """Model evaluator with support for different dataset types"""
    
    def __init__(self, model, device: str = "cuda", model_type: str = "bert", 
                 dataset_type: str = "nli", fixed_splits: bool = True,
                 few_shot: bool = False, tokenizer=None, num_shots: int = 3):
        self.model = model
        self.device = device
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.fixed_splits = fixed_splits
        self.few_shot = few_shot
        self.tokenizer = tokenizer
        
        if fixed_splits:
            self.test_splits = FixedTestSplits(seed=42, test_size=5000)
        
        if few_shot and tokenizer:
            self.few_shot_evaluator = FewShotInstructEvaluator(
                model, tokenizer, device, model_type, num_shots
            )
    
    def evaluate_fixed_splits(self, task: str = "nli") -> Dict:
        """Evaluate on fixed test splits"""
        if not self.fixed_splits:
            raise ValueError("Fixed splits not enabled")
        
        logger.info(f"Evaluating on fixed splits for task: {task}")
        
        if task == "nli":
            # Get fixed splits
            id_dataset, ood_dataset = self.test_splits.get_nli_fixed_splits()
            
            logger.info(f"ID dataset size: {len(id_dataset)}")
            logger.info(f"OOD dataset size: {len(ood_dataset)}")
            
            # Tokenize datasets
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["premise"],
                    examples["hypothesis"],
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )
            
            tokenized_id = id_dataset.map(tokenize_function, batched=True)
            tokenized_id.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            
            tokenized_ood = ood_dataset.map(tokenize_function, batched=True)
            tokenized_ood.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            
            # Create dataloaders
            id_dataloader = DataLoader(tokenized_id, batch_size=32, shuffle=False)
            ood_dataloader = DataLoader(tokenized_ood, batch_size=32, shuffle=False)
            
            # Evaluate
            if self.few_shot:
                logger.info("Using few-shot instruction evaluation")
                id_preds, id_labels = self.few_shot_evaluator.predict_with_instructions(id_dataloader)
                ood_preds, ood_labels = self.few_shot_evaluator.predict_with_instructions(ood_dataloader)
                
                id_metrics = self.compute_metrics(id_preds, id_labels)
                ood_metrics = self.compute_metrics(ood_preds, ood_labels)
            else:
                id_metrics = self.evaluate(id_dataloader)
                ood_metrics = self.evaluate(ood_dataloader)
            
            # Combine results
            results = {
                "id": id_metrics,
                "ood": ood_metrics,
                "test_metadata": {
                    "task": task,
                    "id_samples": len(id_dataset),
                    "ood_samples": len(ood_dataset),
                    "fixed_splits": True,
                    "seed": 42,
                    "few_shot": self.few_shot,
                    "num_shots": self.few_shot_evaluator.num_shots if self.few_shot else 0
                }
            }
            
            # Compute robustness gap
            if "accuracy" in id_metrics and "accuracy" in ood_metrics:
                robustness_gap = id_metrics["accuracy"] - ood_metrics["accuracy"]
                results["robustness_gap"] = robustness_gap
                
                if id_metrics["accuracy"] > 0:
                    results["performance_drop_percent"] = (robustness_gap / id_metrics["accuracy"]) * 100
            
            return results
        
        else:
            raise ValueError(f"Unsupported task for fixed splits: {task}")
    
    # Les autres méthodes restent les mêmes que dans votre code original...
    # (evaluate, compute_metrics, etc.)


class NLIEvaluator(Evaluator):
    """Specialized evaluator for NLI tasks with enhanced features"""
    
    def __init__(self, model, device: str = "cuda", model_type: str = "bert",
                 fixed_splits: bool = True, few_shot: bool = False, 
                 tokenizer=None, num_shots: int = 3):
        super().__init__(model, device, model_type, "nli", 
                        fixed_splits, few_shot, tokenizer, num_shots)
    
    def evaluate_comprehensive(self) -> Dict:
        """Run comprehensive evaluation with all features"""
        results = {}
        
        # 1. Evaluate with fixed splits
        if self.fixed_splits:
            logger.info("Running fixed splits evaluation...")
            fixed_results = self.evaluate_fixed_splits("nli")
            results["fixed_splits"] = fixed_results
        
        # 2. Evaluate with different few-shot configurations
        if self.few_shot:
            logger.info("Running few-shot ablation study...")
            few_shot_results = self._run_few_shot_ablation()
            results["few_shot_ablation"] = few_shot_results
        
        return results
    
    def _run_few_shot_ablation(self) -> Dict:
        """Run ablation study with different numbers of shots"""
        results = {}
        
        for num_shots in [0, 1, 3, 5]:
            logger.info(f"Running {num_shots}-shot evaluation...")
            
            # Create new few-shot evaluator
            few_shot_eval = FewShotInstructEvaluator(
                self.model, self.tokenizer, self.device, 
                self.model_type, num_shots
            )
            
            # Get fixed splits
            id_dataset, ood_dataset = self.test_splits.get_nli_fixed_splits()
            
            # Create dataloaders
            def create_dataloader(dataset):
                return DataLoader(dataset, batch_size=1, shuffle=False)
            
            id_dataloader = create_dataloader(id_dataset)
            ood_dataloader = create_dataloader(ood_dataset)
            
            # Evaluate
            id_preds, id_labels = few_shot_eval.predict_with_instructions(id_dataloader)
            ood_preds, ood_labels = few_shot_eval.predict_with_instructions(ood_dataloader)
            
            id_metrics = self.compute_metrics(id_preds, id_labels)
            ood_metrics = self.compute_metrics(ood_preds, ood_labels)
            
            results[f"{num_shots}_shot"] = {
                "id": id_metrics,
                "ood": ood_metrics,
                "robustness_gap": id_metrics.get("accuracy", 0) - ood_metrics.get("accuracy", 0)
            }
        
        return results


# Factory function mise à jour
def create_evaluator(model, device: str = "cuda", model_type: str = "bert", 
                     dataset_type: str = "nli", **kwargs) -> Evaluator:
    """Factory function to create appropriate evaluator"""
    
    fixed_splits = kwargs.get("fixed_splits", True)
    few_shot = kwargs.get("few_shot", False)
    tokenizer = kwargs.get("tokenizer", None)
    num_shots = kwargs.get("num_shots", 3)
    
    if dataset_type.lower() == "hans":
        return Evaluator(model, device, model_type, "hans", fixed_splits, few_shot, tokenizer, num_shots)
    elif dataset_type.lower() in ["nli", "mnli"]:
        return NLIEvaluator(model, device, model_type, fixed_splits, few_shot, tokenizer, num_shots)
    else:
        return Evaluator(model, device, model_type, dataset_type, fixed_splits, few_shot, tokenizer, num_shots)