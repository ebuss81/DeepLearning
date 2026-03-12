#!/usr/bin/env bash
set -e

# -----------------------------
# 0) System deps (Ubuntu 22.04)
# -----------------------------
apt-get update
apt-get install -y python3-venv python3-dev build-essential

# -----------------------------
# 1) Create + activate venv
# -----------------------------
python3 -m venv /opt/mamba_env
source /opt/mamba_env/bin/activate

# -----------------------------
# 2) Base Python tooling
# -----------------------------
pip install -U pip setuptools wheel packaging

# -----------------------------
# 3) Install PyTorch (CUDA 12.1)
# -----------------------------
pip install \
  torch==2.3.0 \
  torchvision==0.18.0 \
  torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121

# -----------------------------
# 4) Build helpers
# -----------------------------
pip install ninja einops

# -----------------------------
# 5) Transformers
# -----------------------------
pip install "transformers>=4.38,<5"

# -----------------------------
# 6) mamba-ssm
# -----------------------------
pip install mamba-ssm --no-build-isolation

# -----------------------------
# 7) Extra packages
# -----------------------------
pip install optuna scikit-learn pandas pytorch_lightning

# -----------------------------
# 8) Sanity check
# -----------------------------
python - << 'EOF'
import platform
import torch
import transformers

print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print("selective_scan_fn import OK")
EOF

echo "Environment setup complete."
