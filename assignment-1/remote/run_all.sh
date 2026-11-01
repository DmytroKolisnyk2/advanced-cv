#!/usr/bin/env bash
# Every GPU experiment in order; each writes results/exps/<name>/ and is skipped when already done.
set -uo pipefail
cd /workspace/a1
export PATH=$HOME/.local/bin:$PATH HF_HOME=/root/hf UV_PROJECT_ENVIRONMENT=/root/venv UV_CACHE_DIR=/root/uvcache PYTHONUNBUFFERED=1
step() { echo "=== $(date -u +%H:%M:%S) $*"; uv run python "$@" || echo "!!! FAILED: $*"; }
step scripts/prepare.py
step scripts/baselines.py
step scripts/run.py --family cnn
step scripts/run.py --family vit
step scripts/run.py --family ssl
step scripts/interpret.py
step scripts/ensemble.py
step scripts/report.py
echo "=== $(date -u +%H:%M:%S) ALL DONE"
