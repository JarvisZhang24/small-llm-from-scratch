#!/usr/bin/env bash
# Single-command JarvisLM-350M run. Requires one NVIDIA H100 and about 25 GB
# of free disk space for 10B GPT-2 uint16 tokens plus checkpoints.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m jarvislm.training.train --prepare-data --wandb "$@"
