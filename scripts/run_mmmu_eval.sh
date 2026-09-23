#!/usr/bin/env bash
# One-command MMMU-val eval: vLLM inference once, then score the same responses with both parsers.
# Usage: bash scripts/run_mmmu_eval.sh --model_path <HF id or ckpt dir> --data_root <HF datasets cache> --output_dir <dir> [infer.py args...]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DIR=outputs/qwen3vl4b_mmmu_val
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

python "$DIR/infer.py" --output "$OUTPUT_DIR/predictions.jsonl" "${ARGS[@]+"${ARGS[@]}"}"
for parser in qwen mmmu; do
  python "$DIR/score.py" --pred "$OUTPUT_DIR/predictions.jsonl" --parser "$parser" --output_dir "$OUTPUT_DIR"
done
