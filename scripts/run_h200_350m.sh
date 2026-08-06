#!/usr/bin/env bash
# Compatibility wrapper for the persistent RunPod H200 workflow.
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "This wrapper takes no CLI arguments; configure scripts/runpod/run_350m.sh with environment variables." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$repo_root/scripts/runpod/run_350m.sh" full
