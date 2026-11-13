#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="$PWD/.matplotlib"
OUT="${1:-results/counterbalanced-reproduced}"
"$PYTHON" -m inside_transformer.counterbalanced --output "$OUT"
"$PYTHON" -m inside_transformer.counterbalanced_analyze --output "$OUT"
"$PYTHON" scripts/audit_results.py --extension "$OUT" --output "$OUT/numeric-audit.json"
"$PYTHON" -m pytest -q
