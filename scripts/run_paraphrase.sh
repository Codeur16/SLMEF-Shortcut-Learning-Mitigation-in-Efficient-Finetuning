#!/bin/bash
# Run all paraphrase detection experiments

set -e

echo "Running paraphrase detection experiments..."

python scripts/run_all_experiments.py \
    --models bert roberta flant5 llama mistral \
    --tasks paraphrase \
    --exp_types base_eval qlora_ft rgpeft \
    --output_dir experiments/paraphrase

echo "Paraphrase detection experiments completed!"
