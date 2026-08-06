#!/usr/bin/env bash
# Deprecated wrapper: this project now uses the persistent RunPod H200 workflow.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "This project now targets H200. Use $repo_root/scripts/run_h200_350m.sh." >&2
exit 2
