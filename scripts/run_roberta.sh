#!/bin/bash
# Run all experiments for RoBERTa

set -e

echo "Running RoBERTa experiments..."

python scripts/run_all_experiments.py \
    --models roberta \
    --tasks nli sentiment qa paraphrase \
    --exp_types base_eval qlora_ft rgpeft \
    --output_dir experiments/roberta

echo "RoBERTa experiments completed!"