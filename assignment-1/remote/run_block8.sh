#!/usr/bin/env bash
# Block 8: test-time harmonisation (no retraining) and soil-mixup runs, then ensembles and tables again.
set -uo pipefail
cd /workspace/a1
export PATH=$HOME/.local/bin:$PATH HF_HOME=/root/hf UV_PROJECT_ENVIRONMENT=/root/venv UV_CACHE_DIR=/root/uvcache PYTHONUNBUFFERED=1
step() { echo "=== $(date -u +%H:%M:%S) $*"; uv run python "$@" || echo "!!! FAILED: $*"; }
step scripts/harmonise.py
step scripts/run.py --only mix_cnn_r18_bs8_grouped_frozenbn,mix_vit_scratch,mix_ssl_dinov3_s_ft
step scripts/ensemble.py
step scripts/report.py
echo "=== $(date -u +%H:%M:%S) ALL DONE"
