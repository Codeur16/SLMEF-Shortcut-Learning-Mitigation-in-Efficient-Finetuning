#!/bin/bash
# Run all experiments for BERT

set -e

echo "Running BERT experiments..."

python scripts/run_all_experiments.py \
    --models bert \
    --tasks nli sentiment qa paraphrase \
    --exp_types base_eval qlora_ft rgpeft \
    --output_dir experiments/bert

echo "BERT experiments completed!"