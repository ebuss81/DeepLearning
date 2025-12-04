#!/bin/bash

ARGS_LIST=(
  "--model mamba --metric loss" 
  "--model CNN1D --metric loss"
#  "--model s4  --metric loss"
  "--model Inception1D --metric loss"
)

for args in "${ARGS_LIST[@]}"; do
    echo "Running python script with args: $args"
    python optuna_general.py $args

    echo "Done: $args"
    echo "=============================================="
done

