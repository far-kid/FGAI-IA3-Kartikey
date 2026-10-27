#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Use: bash scripts/run_cloud.sh [all|prepare|train|evaluate|report] [model]
cd "$(dirname "$0")/.."
stage="${1:-all}"
model="${2:-}"

prepare() { python -m src.prepare_data --train-size 8000 --val-size 500 --test-size 500 --seed 42; }
train_one() {
  case "$1" in
    transformer) python -m src.train --model transformer --epochs 8 --batch-size 32 --grad-accum 1 --seed 42 ;;
    mt5) python -m src.train --model mt5 --epochs 3 --batch-size 8 --grad-accum 4 --seed 42 ;;
    llama) python -m src.train --model llama --epochs 2 --batch-size 2 --grad-accum 16 --seed 42 ;;
    gemma) python -m src.train --model gemma --epochs 2 --batch-size 8 --grad-accum 4 --seed 42 ;;
    *) echo "Unknown model: $1" >&2; exit 2 ;;
  esac
}
evaluate_one() { python -m src.evaluate --model "$1" --temperature-samples 100; }

case "$stage" in
  all)
    prepare
    for name in transformer mt5 llama gemma; do train_one "$name"; evaluate_one "$name"; done
    python -m src.report
    ;;
  prepare) prepare ;;
  train) train_one "$model" ;;
  evaluate) evaluate_one "$model" ;;
  report) python -m src.report ;;
  *) echo "Usage: bash scripts/run_cloud.sh [all|prepare|train|evaluate|report] [model]" >&2; exit 2 ;;
esac
