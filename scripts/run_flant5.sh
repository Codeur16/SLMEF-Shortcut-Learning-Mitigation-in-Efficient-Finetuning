#!/bin/bash
# Run all experiments for Flan-T5

set -e

echo "Running Flan-T5 experiments..."

python scripts/run_all_experiments.py \
    --models flant5 \
    --tasks nli sentiment qa paraphrase \
    --exp_types base_eval qlora_ft rgpeft \
    --output_dir experiments/flant5

echo "Flan-T5 experiments completed!"