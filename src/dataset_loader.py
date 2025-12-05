"""
Module pour le chargement unifié des datasets.
Supporte : nli, nlu, qa, si
"""
from datasets import load_dataset as hf_load_dataset
from typing import Tuple, Dict, Any

def load_nli_dataset() -> Tuple[Any, Any]:
    """
    Charge le dataset NLI (Natural Language Inference)
    Utilise le dataset GLUE/MNLI par défaut
    """
    try:
        # Chargement du dataset MNLI depuis Hugging Face
        dataset = hf_load_dataset('glue', 'mnli')
        return dataset['train'], dataset['validation_matched']
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du dataset NLI: {str(e)}")

def load_nlu_dataset() -> Tuple[Any, Any]:
    """
    Charge un dataset de compréhension du langage naturel
    Utilise ATIS (Airline Travel Information System) par défaut
    """
    try:
        dataset = hf_load_dataset('snips_built_in_intents')
        # Séparation train/val (80/20)
        dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
        return dataset['train'], dataset['test']
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du dataset NLU: {str(e)}")

def load_qa_dataset() -> Tuple[Any, Any]:
    """
    Charge un dataset de questions-réponses
    Utilise SQuAD v2 par défaut
    """
    try:
        dataset = hf_load_dataset('squad_v2')
        return dataset['train'], dataset['validation']
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du dataset QA: {str(e)}")

def load_si_dataset() -> Tuple[Any, Any]:
    """
    Charge un dataset de similarité sémantique
    Utilise le dataset STSB (Semantic Textual Similarity Benchmark)
    """
    try:
        dataset = hf_load_dataset('glue', 'stsb')
        # Séparation train/val (80/20)
        dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
        return dataset['train'], dataset['test']
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du dataset SI: {str(e)}")

def load_dataset(task: str) -> Tuple[Any, Any]:
    """
    Fonction unifiée pour charger n'importe quel dataset.
    
    Args:
        task (str): Type de tâche ('nli', 'nlu', 'qa', 'si')
    
    Returns:
        tuple: (train_dataset, val_dataset)
    
    Raises:
        ValueError: Si la tâche n'est pas supportée
        RuntimeError: En cas d'erreur lors du chargement du dataset
    """
    loaders = {
        'nli': load_nli_dataset,
        'nlu': load_nlu_dataset,
        'qa': load_qa_dataset,
        'si': load_si_dataset
    }
    
    if task not in loaders:
        raise ValueError(
            f"Tâche non supportée : {task}. "
            f"Choisissez parmi : {list(loaders.keys())}"
        )
    
    return loaders[task]()

# # Exemple d'utilisation
# if __name__ == "__main__":
#     try:
#         # Exemple de chargement du dataset NLI
#         train, val = load_dataset('nli')
#         print("Dataset NLI chargé avec succès !")
#         print(f"Train: {len(train)} exemples")
#         print(f"Validation: {len(val)} exemples")
#     except Exception as e:
#         print(f"Erreur: {str(e)}")