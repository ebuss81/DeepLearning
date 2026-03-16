#!/bin/bash

ARGS_LIST=(
  "--model mamba --metric acc --time_horizon 30min_3classes" # stil optrimising for los but printing acc
  "--model CNN1D --metric acc --time_horizon 30min_3classes"
  "--model Inception1D --metric acc --time_horizon 30min_3classes"
)

for args in "${ARGS_LIST[@]}"; do
    echo "Running python script with args: $args"
    python optuna_general.py $args

    echo "Done: $args"
    echo "=============================================="
done

