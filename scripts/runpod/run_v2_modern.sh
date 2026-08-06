#!/usr/bin/env bash
# Persistent RunPod workflow for the final V2 modern single-GPU recipe.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/runpod/run_v2_modern.sh {preflight|full}

Environment (all optional):
  JARVISLM_VOLUME_ROOT=/workspace/jarvislm-350m
  JARVISLM_USE_WANDB=1
  JARVISLM_V2_PREFLIGHT_STEPS=200
  JARVISLM_V2_STEPS=10500
EOF
}

command=${1:-}
if [[ $# -ne 1 || ! "$command" =~ ^(preflight|full)$ ]]; then
  usage >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
volume_root="${JARVISLM_VOLUME_ROOT:-/workspace/jarvislm-350m}"
data_root="$volume_root/data/fineweb-edu-v1-sample-10bt"
train_dir="$data_root/train"
val_dir="$data_root/val"
manifest="$data_root/manifest.json"
cache_root="$volume_root/cache"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$cache_root/huggingface}"
export WANDB_DIR="${WANDB_DIR:-$cache_root/wandb}"
mkdir -p "$cache_root"

python_bin="${PYTHON:-python}"
verify_data=(
  "$python_bin" "$repo_root/scripts/runpod/verify_prepared_data.py"
  --data-dir "$train_dir" --val-dir "$val_dir" --manifest "$manifest"
)

wandb_args=(--no-wandb)
environment_args=(--no-use-wandb)
if [[ "${JARVISLM_USE_WANDB:-1}" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_TOKEN:-}" ]]; then
    echo "Set WANDB_API_KEY (or WANDB_TOKEN), or set JARVISLM_USE_WANDB=0." >&2
    exit 1
  fi
  wandb_args=(--wandb)
  environment_args=(--use-wandb)
fi

case "$command" in
  preflight)
    "$python_bin" "$repo_root/scripts/runpod/check_environment.py" \
      --stage preflight "${environment_args[@]}"
    "${verify_data[@]}"
    preflight_root="$volume_root/runs/v2-modern-a100-preflight-mb16"
    preflight_steps="${JARVISLM_V2_PREFLIGHT_STEPS:-200}"
    mkdir -p "$preflight_root"
    set -o pipefail
    "$python_bin" -m jarvislm.training.train \
      --recipe v2_modern --run-name jarvislm-v2-modern-a100-preflight \
      --data-dir "$train_dir" --val-dir "$val_dir" \
      --checkpoint-dir "$preflight_root/checkpoints" \
      --max-steps "$preflight_steps" \
      --micro-batch-size 16 --grad-accumulation-steps 32 --num-workers 4 \
      --log-interval 10 --eval-interval 100 --save-interval 100 \
      --required-gpu A100 --compile --resume --muon --ema --eval-use-ema \
      "${wandb_args[@]}" 2>&1 | tee -a "$preflight_root/train.log"
    "$python_bin" "$repo_root/scripts/runpod/assess_preflight.py" \
      --log "$preflight_root/train.log" \
      --checkpoint "$preflight_root/checkpoints/last.pt" \
      --expected-step "$preflight_steps" \
      --report "$preflight_root/preflight_report.json"
    ;;
  full)
    "$python_bin" "$repo_root/scripts/runpod/check_environment.py" \
      --stage full "${environment_args[@]}"
    "${verify_data[@]}"
    "$python_bin" "$repo_root/scripts/runpod/require_preflight.py" \
      --report "$volume_root/runs/v2-modern-a100-preflight-mb16/preflight_report.json" \
      --expected-step "${JARVISLM_V2_PREFLIGHT_STEPS:-200}"
    full_root="$volume_root/runs/v2-modern-h200-5.5b"
    mkdir -p "$full_root"
    "$python_bin" -m jarvislm.training.train \
      --recipe v2_modern --run-name jarvislm-v2-modern-315m-h200-5.5b \
      --data-dir "$train_dir" --val-dir "$val_dir" \
      --checkpoint-dir "$full_root/checkpoints" \
      --max-steps "${JARVISLM_V2_STEPS:-10500}" \
      --micro-batch-size 32 --grad-accumulation-steps 16 --num-workers 8 \
      --log-interval 10 --eval-interval 500 --save-interval 1000 \
      --required-gpu H200 --compile --resume --muon --ema --eval-use-ema \
      "${wandb_args[@]}" 2>&1 | tee -a "$full_root/train.log"
    ;;
esac
