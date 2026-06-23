#!/usr/bin/env bash
# Full FP8-experts verification. Uses the GPU — only run with explicit go-ahead.
#
#   bash tests/verify_fp8_full.sh [BF16_CKPT] [FP8_OUT_DIR]
#
# Defaults: BF16_CKPT=Zyphra/ZONOS2, FP8_OUT_DIR=models/zonos2-fp8
set -euo pipefail

BF16="${1:-Zyphra/ZONOS2}"
FP8_DIR="${2:-models/zonos2-fp8}"
PY="${PYTHON:-uv run python}"
export PYTHONPATH=python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$(dirname "$0")/.."

echo "### 1/4 Isolated FP8 kernel numeric check (GPU)"
$PY tests/test_fp8_experts_numeric.py

echo "### 2/4 Convert real checkpoint -> FP8 (CPU/IO heavy)"
if [ ! -f "$FP8_DIR/model.pth" ]; then
  $PY models/quantize_fp8.py --in "$BF16" --out "$FP8_DIR"
else
  echo "  (reusing existing $FP8_DIR/model.pth)"
fi

echo "### 3/4 End-to-end TTS: fp8 vs bf16 (GPU)"
$PY tests/verify_fp8_e2e.py --bf16 "$BF16" --fp8 "$FP8_DIR" --out /tmp/zonos2_fp8_check

echo "### 4/4 Regression: existing tests with FP8 path untouched (default bf16)"
$PY -m pytest -o addopts="" tests/ -q -k "not fp8" || true

echo "Done. Inspect /tmp/zonos2_fp8_check/*.wav and the memory/cosine report above."
