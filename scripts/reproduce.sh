#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export MPLCONFIGDIR="$PWD/.matplotlib"
export HF_HUB_DISABLE_XET=1
OUT="${1:-results/reproduced}"
if [ -e "$OUT" ]; then
  echo "Output already exists; choose a new directory." >&2
  exit 1
fi
"$PYTHON" -m inside_transformer.experiment --output "$OUT"
"$PYTHON" -m inside_transformer.analyze --output "$OUT"
"$PYTHON" scripts/audit_results.py --original "$OUT" --output "$OUT/numeric-audit.json"
"$PYTHON" -m pytest -q
