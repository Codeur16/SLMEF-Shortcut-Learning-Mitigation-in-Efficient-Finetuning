#!/bin/bash
# Run all experiments for LLaMA

set -e

echo "Running LLaMA experiments..."
echo "Note: Requires HuggingFace token for LLaMA access"

# Set HuggingFace token if available
if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
fi

python scripts/run_all_experiments.py \
    --models llama \
    --tasks nli sentiment qa paraphrase \
    --exp_types base_eval qlora_ft rgpeft \
    --output_dir experiments/llama \
    --max_workers 1  # Reduce workers for memory-intensive models

echo "LLaMA experiments completed!"