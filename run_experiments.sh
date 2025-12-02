#!/bin/bash

# -------------------------
# LISTES PAR DEFAUT
# -------------------------
ALL_TYPES=("efficient-finetuning" "neusy-finetuning")
ALL_TASKS=("nli" "nlu" "qa" "si")
ALL_MODELS=("flant5" "llama" "bert" "mistral" "gpt")

# -------------------------
# PARSE DES ARGUMENTS
# -------------------------
SELECTED_TYPE=""
SELECTED_TASK=""
SELECTED_MODEL=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --type) SELECTED_TYPE="$2"; shift ;;
        --task) SELECTED_TASK="$2"; shift ;;
        --model) SELECTED_MODEL="$2"; shift ;;
        *) echo "Argument inconnu: $1"; exit 1 ;;
    esac
    shift
done

# -------------------------
# GENERATION DES LISTES FILTREES
# -------------------------
if [[ -n "$SELECTED_TYPE" ]]; then
    TYPES=("$SELECTED_TYPE")
else
    TYPES=("${ALL_TYPES[@]}")
fi

if [[ -n "$SELECTED_TASK" ]]; then
    TASKS=("$SELECTED_TASK")
else
    TASKS=("${ALL_TASKS[@]}")
fi

if [[ -n "$SELECTED_MODEL" ]]; then
    MODELS=("$SELECTED_MODEL")
else
    MODELS=("${ALL_MODELS[@]}")
fi

# -------------------------
# EXECUTION
# -------------------------
echo "=== EXPERIMENTS LANCÉS ==="
echo "Types  : ${TYPES[*]}"
echo "Tâches : ${TASKS[*]}"
echo "Modèles: ${MODELS[*]}"
echo "==================================="

for type in "${TYPES[@]}"; do
  for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do

      echo "============================================"
      echo "   Lancement : $type | $task | $model"
      echo "============================================"

      # ========= TRAIN ==========
      python3 src/train.py \
        --model "$model" \
        --task "$task" \
        --exp_type "$type" \
        --save_dir "outputs/checkpoints/${type}/${task}/${model}"

      if [[ $? -ne 0 ]]; then
          echo "Erreur pendant l'entraînement."
          continue
      fi

      # ========= INFERENCE ==========
      python3 src/inference.py \
        --model "$model" \
        --task "$task" \
        --exp_type "$type" \
        --checkpoint "outputs/checkpoints/${type}/${task}/${model}" \
        --output_file "outputs/predictions/${type}_${task}_${model}.json"

      # ========= EVALUATION ==========
      python3 src/evaluate.py \
        --task "$task" \
        --predictions "outputs/predictions/${type}_${task}_${model}.json" \
        --metrics_file "outputs/results/${type}_${task}_${model}.csv"

    done
  done
done

echo "==== TOUS LES EXPÉRIMENTS SONT TERMINÉS ===="