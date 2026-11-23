#!/usr/bin/env bash
# MMMU-Pro standard (10 options), test split (1,730 questions): baseline vs v1b in one command.
#   1) vLLM inference per model (infer_standard10.py)  2) scripts/score.py with both parsers  3) compare.py
#
# Usage: bash mmmu_pro_eval/run_compare.sh --data_root <HF datasets cache> [--out_root DIR] [--only baseline|v1b]
#        [--v1b_path <HF id or local merged dir>] [--image_layout interleave|front] [--limit N] [infer args...]
# Extra args (e.g. --max_new_tokens 8192, --gpu_memory_utilization 0.85) go to both inference runs.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$DIR/../scripts"

OUT_ROOT=outputs/mmmu_pro_std10
BASELINE_PATH=Qwen/Qwen3-VL-4B-Instruct               # revision pinned by infer.py (MODEL_REVISION)
V1B_PATH=yujin16/qwen3vl4b-mmmu-v1b                    # RFT + DPO v1b, LoRA merged (docs/rft_dpo_results.md)
ONLY=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --v1b_path) V1B_PATH="$2"; shift 2 ;;
    --baseline_path) BASELINE_PATH="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

run_one() {  # name model_path revision
  local name="$1" path="$2" rev="$3" out="$OUT_ROOT/$1"
  if [[ -f "$out/predictions.jsonl" ]]; then
    echo "[$name] $out/predictions.jsonl exists, skipping inference (delete it to rerun)"
  else
    python "$DIR/infer_standard10.py" --model_path "$path" --model_revision "$rev" \
      --output "$out/predictions.jsonl" "${ARGS[@]+"${ARGS[@]}"}"
  fi
  for parser in qwen mmmu; do
    for extract in final none; do
      python "$SCRIPTS/score.py" --pred "$out/predictions.jsonl" --parser "$parser" --extract "$extract" \
        --output_dir "$out" > /dev/null
    done
  done
  echo "[$name] scored -> $out/scores_*.{json,md,csv}"
}

# baseline: pinned HF revision (infer.py default). v1b: '' = no revision pin (HF main, or a local dir).
[[ -z "$ONLY" || "$ONLY" == baseline ]] && run_one baseline "$BASELINE_PATH" ebb281ec70b05090aa6165b016eac8ec08e71b17
[[ -z "$ONLY" || "$ONLY" == v1b ]] && run_one v1b "$V1B_PATH" ''

if [[ -f "$OUT_ROOT/baseline/scores_qwen.csv" && -f "$OUT_ROOT/v1b/scores_qwen.csv" ]]; then
  python "$DIR/compare.py" --a "$OUT_ROOT/baseline" --b "$OUT_ROOT/v1b" --name_a baseline --name_b v1b \
    --out "$OUT_ROOT/compare"
fi
