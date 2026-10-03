#!/usr/bin/env bash
# Ablation: DeepStack mergers (full FT) vs LLM LoRA vs both, vision encoder frozen in all arms.
# Trains the three arms with train_ft.py, evaluates baseline + arms with run_mmmu_eval.sh (unchanged eval pipeline),
# then writes the comparison table.
# Usage: bash scripts/run_ablation.sh --train_data <jsonl> --image_root <dir> --data_root <MMMU HF cache> \
#          [--out_root outputs/ablation] [--arms "deepstack lora both"] [--skip_baseline] [--skip_eval] [train_ft.py args...]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_ROOT=outputs/ablation
ARMS="deepstack lora both"
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct
TRAIN_DATA=; IMAGE_ROOT=; DATA_ROOT=; SKIP_BASELINE=0; SKIP_EVAL=0
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --train_data) TRAIN_DATA="$2"; shift 2 ;;
    --image_root) IMAGE_ROOT="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --skip_baseline) SKIP_BASELINE=1; shift ;;
    --skip_eval) SKIP_EVAL=1; shift ;;
    *) TRAIN_ARGS+=("$1"); shift ;;
  esac
done
[[ -n "$TRAIN_DATA" ]] || { echo "--train_data is required" >&2; exit 1; }

arm_flags() {
  case "$1" in
    deepstack) echo "--tune_deepstack" ;;
    lora) echo "--lora_llm" ;;
    both) echo "--tune_deepstack --lora_llm" ;;
    *) echo "unknown arm: $1" >&2; exit 1 ;;
  esac
}

EVAL_ARGS=(); [[ -n "$DATA_ROOT" ]] && EVAL_ARGS+=(--data_root "$DATA_ROOT")

for arm in $ARMS; do
  # shellcheck disable=SC2046
  python "$DIR/train_ft.py" --model_path "$MODEL_PATH" --train_data "$TRAIN_DATA" --image_root "$IMAGE_ROOT" \
    --output_dir "$OUT_ROOT/ckpt_$arm" $(arm_flags "$arm") "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"
done

[[ "$SKIP_EVAL" == 1 ]] && exit 0

RUNS=()
if [[ "$SKIP_BASELINE" == 0 ]]; then
  bash "$DIR/run_mmmu_eval.sh" --model_path "$MODEL_PATH" --output_dir "$OUT_ROOT/eval_baseline" "${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}"
  RUNS+=("baseline=$OUT_ROOT/eval_baseline")
fi
for arm in $ARMS; do
  bash "$DIR/run_mmmu_eval.sh" --model_path "$OUT_ROOT/ckpt_$arm" --model_revision '' \
    --output_dir "$OUT_ROOT/eval_$arm" "${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}"
  RUNS+=("$arm=$OUT_ROOT/eval_$arm")
done
python "$DIR/compare_ablation.py" --runs "${RUNS[@]}" --output "$OUT_ROOT/comparison.md"
