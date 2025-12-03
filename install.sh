#!/usr/bin/env bash
set -e

# Install venv support
apt-get update
apt-get install -y python3-venv python3-dev build-essential

# Create & activate venv
python3 -m venv /opt/mamba_env
source /opt/mamba_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch 2.3.0 (CUDA 12.1)
pip install \
  "torch==2.3.0" "torchvision==0.18.0" "torchaudio==2.3.0" \
  --index-url https://download.pytorch.org/whl/cu121

# Build deps needed for mamba-ssm
pip install packaging ninja einops transformers

# Install mamba-ssm
pip install "mamba-ssm[causal-conv1d]" --no-build-isolation

# Extra ML libraries you requested
pip install optuna scikit-learn pandas pytorch-lightning

echo "=== DONE ==="
echo "Activate the venv with:"
echo "source /opt/mamba_env/bin/activate"