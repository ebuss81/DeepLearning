#!/bin/bash

ARGS_LIST=(
#  "--model mamba --metric acc" # stil optrimising for los but printing acc
#  "--model CNN1D --metric acc"
#  "--model s4  --metric acc"
  "--model Inception1D --metric acc"
)

for args in "${ARGS_LIST[@]}"; do
    echo "Running python script with args: $args"
    python optuna_general.py $args

    echo "Done: $args"
    echo "=============================================="
done

