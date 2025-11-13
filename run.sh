#!/bin/bash

ARGS_LIST=(
  "--model Inception1D --metric acc"
  "--model CNN1D --metric acc"
  "--model s4  --metric acc"
)

for args in "${ARGS_LIST[@]}"; do
    echo "Running python script with args: $args"
    python optuna_general.py $args

    echo "Done: $args"
    echo "=============================================="
done

