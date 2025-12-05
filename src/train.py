# Train code

import argparse
from models.flant5 import FlanT5Model
def load_dataset_for_task(task):
    """Charge le dataset approprié en fonction de la tâche."""
    from datasets import Dataset
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    try:
        if task == 'nli':
            # Chargement du dataset depuis le fichier CSV local
            df = pd.read_csv('data/balanced_10k_dataset_v2_enriched.csv')
            
            # Conversion des colonnes en format attendu
            # Remplacez 'premise', 'hypothesis' et 'label' par les noms de vos colonnes
            df = df.rename(columns={
                'premise': 'premise',         # Colonne contenant les prémisses
                'hypothesis': 'hypothesis',   # Colonne contenant les hypothèses
                'label': 'label'              # Colonne contenant les labels
            })
            
            # Séparation en train (90%) et validation (10%)
            train_df, val_df = train_test_split(
                df, 
                test_size=0.1, 
                random_state=42,
                stratify=df['label']
            )
            
            # Conversion des DataFrames en datasets Hugging Face
            train_ds = Dataset.from_pandas(train_df)
            val_ds = Dataset.from_pandas(val_df)
            
            return train_ds, val_ds
            
        # [Le reste du code reste inchangé]
        
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du dataset: {str(e)}")

def load_dataset_for_task2(task):
    """Charge le dataset approprié en fonction de la tâche."""
    from datasets import load_dataset as hf_load_dataset
    
    try:
        if task == 'nli':
            # Chargement du dataset NLI (MultiNLI)
            dataset = hf_load_dataset('glue', 'mnli')
            return dataset['train'], dataset['validation_matched']
        
        elif task == 'nlu':
            # Chargement d'un dataset de compréhension de langage
            dataset = hf_load_dataset('snips_built_in_intents')
            # Séparation train/val (80/20)
            dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
            return dataset['train'], dataset['test']
        
        elif task == 'qa':
            # Chargement d'un dataset de questions-réponses
            dataset = hf_load_dataset('squad_v2')
            return dataset['train'], dataset['validation']
        
        elif task == 'si':
            # Chargement d'un dataset de similarité sémantique
            dataset = hf_load_dataset('glue', 'stsb')
            # Séparation train/val (80/20)
            dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
            return dataset['train'], dataset['test']
        
        else:
            raise ValueError(f"Tâche non supportée: {task}")
            
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du dataset: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Architecture du modèle (flant5, llama, bert, mistral, gpt).")
    parser.add_argument("--task", required=True, help="Tâche à effectuer (nli, nlu, qa, si).")
    parser.add_argument("--exp_type", required=True, help="Type d'expérience (efficient-finetuning, neusy-finetuning).")
    parser.add_argument("--save_dir", required=True, help="Dossier de sauvegarde des modèles.")
    args = parser.parse_args()

    print(f"Paramètres: model={args.model}, task={args.task}, exp_type={args.exp_type}, save_dir={args.save_dir}")

    # Initialisation du modèle
    if args.model == 'flant5':
        model = FlanT5Model(model_name=args.model, exp_type=args.exp_type)
    else:
        raise ValueError(f"Modèle non supporté: {args.model}")

    # Chargement du dataset
    try:
        train_ds, val_ds = load_dataset_for_task(args.task)
        print(f"Dataset {args.task} chargé avec succès")
        print(f"  - Exemples d'entraînement: {len(train_ds)}")
        print(f"  - Exemples de validation: {len(val_ds)}")
    except Exception as e:
        print(f"Erreur lors du chargement du dataset: {str(e)}")
        return

    # Préparation et entraînement du modèle
    try:
        model.load()
        model.prepare_for_training()
        model.train(train_ds, val_ds, args.save_dir)
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {str(e)}")
        return

if __name__ == "__main__":
    main()