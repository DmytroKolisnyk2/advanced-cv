#!/usr/bin/env bash
# Talk to the pod tracked in remote/.pod.json (runpod_ctl.py create, then runpod_ctl.py ssh).
#   remote/pod.sh push            code -> /workspace/a1 (no data, caches, checkpoints)
#   remote/pod.sh pull            results/ (+ logs) back
#   remote/pod.sh run CMD...      run a command in /workspace/a1
#   remote/pod.sh bg NAME CMD...  run detached, log to runs/NAME.log
#   remote/pod.sh log NAME [N]    tail a detached job's log
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
read -r IP PORT < <(python3 -c "import json; d = json.load(open('$HERE/.pod.json'))['ssh']; print(d['ip'], d['port'])")
SSH_OPTS=(-p "$PORT" -i "$HOME/.ssh/runpod_a_nlp" -o StrictHostKeyChecking=accept-new
          -o UserKnownHostsFile="$HERE/.known_hosts" -o ServerAliveInterval=30 -o ConnectTimeout=20)
REMOTE=/workspace/a1
ENV="export PATH=\$HOME/.local/bin:\$PATH HF_HOME=/root/hf UV_PROJECT_ENVIRONMENT=/root/venv UV_CACHE_DIR=/root/uvcache PYTHONUNBUFFERED=1; set -a; [ -f .env ] && . ./.env; set +a"

case "$1" in
  push)
    rsync -az --no-owner --no-group --info=stats1 -e "ssh ${SSH_OPTS[*]}" \
      --exclude .venv --exclude data --exclude cache --exclude checkpoints --exclude runs --exclude __pycache__ \
      --exclude 'remote/.pod.json' --exclude 'remote/.known_hosts' --exclude '.git' --exclude .env \
      "$ROOT/" "root@$IP:$REMOTE/"
    # only the Kaggle credentials go to the pod (for the data download), never the RunPod key
    grep -E '^KAGGLE_[A-Z_]+=.+' "$ROOT/.env" | ssh "${SSH_OPTS[@]}" "root@$IP" "cat > $REMOTE/.env" ;;
  pull)
    rsync -az --no-owner --no-group --info=stats1 -e "ssh ${SSH_OPTS[*]}" "root@$IP:$REMOTE/results/" "$ROOT/results/"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "root@$IP:$REMOTE/runs/" "$ROOT/runs/" || true ;;
  pullckpt)
    shift; for n in "$@"; do rsync -az -e "ssh ${SSH_OPTS[*]}" "root@$IP:$REMOTE/checkpoints/$n*" "$ROOT/checkpoints/"; done ;;
  run)
    shift; ssh "${SSH_OPTS[@]}" "root@$IP" "cd $REMOTE && $ENV && $*" ;;
  bg)
    name=$2; shift 2
    ssh "${SSH_OPTS[@]}" "root@$IP" "cd $REMOTE && $ENV && mkdir -p runs && setsid nohup bash -c '$*' > runs/$name.log 2>&1 < /dev/null & echo started $name" ;;
  log)
    ssh "${SSH_OPTS[@]}" "root@$IP" "tail -n ${3:-20} $REMOTE/runs/$2.log" ;;
  *) echo "unknown command $1"; exit 1 ;;
esac
