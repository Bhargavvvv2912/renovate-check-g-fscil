#!/usr/bin/env bash
# Usage: bash scripts/run_cub.sh [gpu_id]
set -euo pipefail
GPU="${1:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" python main.py --config ./exps/cub.json
