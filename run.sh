#!/bin/bash

ARGS_LIST=(
  "--model mamba --metric acc --time_horizon 6h" # stil optrimising for los but printing acc
  "--model CNN1D --metric acc --time_horizon 6h"
  "--model Inception1D --metric acc --time_horizon 6h"
)

for args in "${ARGS_LIST[@]}"; do
    echo "Running python script with args: $args"
    python optuna_general.py $args

    echo "Done: $args"
    echo "=============================================="
done

