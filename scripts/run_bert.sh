#!/bin/bash
# Run all experiments for BERT

set -e

echo "Running BERT experiments..."

python3 scripts/run_all_experiments.py \
    --models bert \
    --tasks nli \
    --exp_types base_eval \
    --output_dir experiments/bert

echo "BERT experiments completed!"