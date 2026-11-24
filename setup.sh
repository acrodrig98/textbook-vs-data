#!/usr/bin/env bash
# Setup for Project 1 (Textbook vs. Data). Run from the repo root on a Linux machine with an NVIDIA GPU.
# Before running this script:
#   1. Make a Hugging Face account and accept the Gemma license at
#      https://huggingface.co/google/gemma-2-2b (Gemma models are gated).
#   2. Create a read token at https://huggingface.co/settings/tokens
set -euo pipefail

# Install uv if it's missing
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.11 .venv
source .venv/bin/activate

# Installs circuit-tracer from the official repo, pinned to a tested commit, along with the
# project's other dependencies. On a GPU machine, the default PyPI torch wheel includes CUDA.
uv pip install -r requirements.txt

# Log in to Hugging Face (paste your token when prompted; it's stored on this machine only)
hf auth login

# Checks that run anywhere, then the GPU check
python -m pytest -q tests/
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "Setup done. Next: python smoke_test.py"

# Week 4: circuit-tracer now loads the Gemma Scope 2 transcoders for Gemma 3 directly,
# through its nnsight backend:
#   ReplacementModel.from_pretrained("google/gemma-3-4b-pt", "<transcoder repo>", backend="nnsight")
# The transcoder repos are listed in the circuit-tracer README. You only need a separate
# environment if you also want SAELens for the residual-stream SAEs.
