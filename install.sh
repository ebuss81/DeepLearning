#!/bin/bash
set -e

VENV=/opt/mamba_env

# OS deps
apt-get update
apt-get install -y python3-venv python3-dev build-essential

# venv (idempotent)
python3 -m venv "$VENV"

# use venv tools directly (no activation)
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

# PyTorch 2.4.1 + CUDA 12.1 (works with causal-conv1d)
"$VENV/bin/pip" install \
  "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1" \
  --index-url https://download.pytorch.org/whl/cu121

# deps + mamba-ssm
"$VENV/bin/pip" install packaging ninja einops transformers
"$VENV/bin/pip" install "mamba-ssm[causal-conv1d]" --no-build-isolation

# extras (optional)
"$VENV/bin/pip" install optuna scikit-learn pandas pytorch-lightning

echo "=== DONE ==="
echo "Activate the venv with:"
echo "source /opt/mamba_env/bin/activate"