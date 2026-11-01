#!/usr/bin/env bash
# One-time pod setup: uv + locked env, competition data (Kaggle API), sanity checks.
set -euo pipefail
cd /workspace/a1
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf
set -a; [ -f .env ] && . ./.env; set +a
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run python -c "import torch, timm; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), timm.__version__)"
if [ ! -d data/raw ] || [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
  mkdir -p data/raw
  uv run kaggle competitions download -c soil-grain-size-from-photos -p data/raw
  (cd data/raw && for z in *.zip; do unzip -q -o "$z" && rm "$z"; done)
fi
du -sh data/raw; find data/raw -maxdepth 2 | head -20
