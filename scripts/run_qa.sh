#!/bin/bash
# Run all QA experiments

set -e

echo "Running QA experiments..."

python scripts/run_all_experiments.py \
    --models bert roberta flant5 llama mistral \
    --tasks qa \
    --exp_types base_eval qlora_ft rgpeft \
    --output_dir experiments/qa

echo "QA experiments completed!"