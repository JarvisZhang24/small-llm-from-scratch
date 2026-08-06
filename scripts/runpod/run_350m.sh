#!/usr/bin/env bash
# Persistent RunPod workflow for the reference V1 350M single-GPU recipe.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/runpod/run_350m.sh {prepare|preflight|full}

Environment (all optional):
  JARVISLM_VOLUME_ROOT=/workspace/jarvislm-350m
  JARVISLM_USE_WANDB=1
  JARVISLM_PREFLIGHT_STEPS=500
  JARVISLM_H200_STEPS=19074
EOF
}

command=${1:-}
if [[ $# -ne 1 || ! "$command" =~ ^(prepare|preflight|full)$ ]]; then
  usage >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
volume_root="${JARVISLM_VOLUME_ROOT:-/workspace/jarvislm-350m}"
data_root="$volume_root/data/fineweb-edu"
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
  prepare)
    "$python_bin" "$repo_root/scripts/runpod/check_environment.py" \
      --stage prepare "${environment_args[@]}"
    if [[ -f "$manifest" ]]; then
      "${verify_data[@]}"
      echo "FineWeb-Edu shards are already complete on the network volume."
      exit 0
    fi
    if [[ -n "$(find "$train_dir" "$val_dir" -type f -name '*.bin' -print -quit 2>/dev/null)" ]]; then
      echo "Found shards without $manifest. Refusing to mix partial data with a new run." >&2
      echo "Inspect or back up the directories before removing them manually." >&2
      exit 1
    fi
    # Some datasets/PyArrow builds can abort while CPython is finalizing worker
    # threads, after every shard was already flushed.  The manifest is the
    # authoritative completion check, so only accept such a non-zero exit when
    # the exact requested token counts can still be validated.
    set +e
    "$python_bin" -m jarvislm.training.train \
      --prepare-data --prepare-only \
      --data-dir "$train_dir" --val-dir "$val_dir"
    prepare_status=$?
    set -e
    if ! "${verify_data[@]}" --write; then
      echo "FineWeb-Edu preparation did not produce a complete valid dataset." >&2
      exit "$prepare_status"
    fi
    if [[ "$prepare_status" -ne 0 ]]; then
      echo "Warning: preparation exited non-zero after valid shards were written; accepted after manifest verification." >&2
    fi
    ;;
  preflight)
    "$python_bin" "$repo_root/scripts/runpod/check_environment.py" \
      --stage preflight "${environment_args[@]}"
    "${verify_data[@]}"
    preflight_root="$volume_root/runs/a100-preflight"
    mkdir -p "$preflight_root"
    preflight_steps="${JARVISLM_PREFLIGHT_STEPS:-500}"
    set -o pipefail
    "$python_bin" -m jarvislm.training.train \
      --run-name jarvislm-350m-a100-preflight \
      --data-dir "$train_dir" --val-dir "$val_dir" \
      --checkpoint-dir "$preflight_root/checkpoints" \
      --max-steps "$preflight_steps" \
      --micro-batch-size 2 --grad-accumulation-steps 256 --num-workers 2 \
      --log-interval 10 --eval-interval 100 --save-interval 100 \
      --required-gpu A100 --compile --resume "${wandb_args[@]}" 2>&1 | tee -a "$preflight_root/train.log"
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
      --report "$volume_root/runs/a100-preflight/preflight_report.json" \
      --expected-step "${JARVISLM_PREFLIGHT_STEPS:-500}"
    full_root="$volume_root/runs/h200-10b"
    mkdir -p "$full_root"
    "$python_bin" -m jarvislm.training.train \
      --run-name jarvislm-350m-h200-10b \
      --data-dir "$train_dir" --val-dir "$val_dir" \
      --checkpoint-dir "$full_root/checkpoints" \
      --max-steps "${JARVISLM_H200_STEPS:-19074}" \
      --micro-batch-size 16 --grad-accumulation-steps 32 --num-workers 4 \
      --log-interval 10 --eval-interval 500 --save-interval 1000 \
      --required-gpu H200 --compile --resume "${wandb_args[@]}"
    ;;
esac
