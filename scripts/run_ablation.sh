#!/usr/bin/env bash
# Ablation: DeepStack mergers (full FT) vs LLM LoRA vs both, vision encoder frozen in all arms.
# Trains the three arms with train_ft.py, evaluates baseline + arms with run_mmmu_eval.sh (unchanged eval pipeline),
# then writes the comparison table.
# Usage: bash scripts/run_ablation.sh --train_data <jsonl> --image_root <dir> --data_root <MMMU HF cache> \
#          [--out_root outputs/ablation] [--arms "deepstack lora both"] [--skip_baseline] [--skip_eval] [--smoke] \
#          [train_ft.py args...]
# --smoke: end-to-end check with the real 4B weights on a small GPU (T4 16GB is the target) before the A100 run:
#   2 training steps per arm with small images, eval on Math only (30 q) with 256 new tokens, fp16 if the GPU has no
#   native bf16. Output goes to outputs/ablation_smoke; its scores mean nothing, it only proves every stage runs.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_ROOT=outputs/ablation
ARMS="deepstack lora both"
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct
TRAIN_DATA=; IMAGE_ROOT=; DATA_ROOT=; SKIP_BASELINE=0; SKIP_EVAL=0; SMOKE=0; OUT_ROOT_SET=0
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_root) OUT_ROOT="$2"; OUT_ROOT_SET=1; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --train_data) TRAIN_DATA="$2"; shift 2 ;;
    --image_root) IMAGE_ROOT="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --skip_baseline) SKIP_BASELINE=1; shift ;;
    --skip_eval) SKIP_EVAL=1; shift ;;
    --smoke) SMOKE=1; shift ;;
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
SMOKE_TRAIN=()
if [[ "$SMOKE" == 1 ]]; then
  [[ "$OUT_ROOT_SET" == 1 ]] || OUT_ROOT=outputs/ablation_smoke
  PIXELS=(--min_pixels $((64 * 32 * 32)) --max_pixels $((256 * 32 * 32)))  # 64-256 visual tokens per image
  # placed before the user's args, so anything passed explicitly still wins
  SMOKE_TRAIN=(--max_steps 2 --warmup_steps 0 --gradient_accumulation_steps 1 --per_device_train_batch_size 1
               --max_length 2048 --save_strategy no --logging_steps 1 --gradient_checkpointing "${PIXELS[@]}")
  DTYPE=$(python -c "import torch; print('bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False) else 'float16')")
  EVAL_ARGS+=(--subjects Math --max_new_tokens 256 --max_model_len 4096 --dtype "$DTYPE" "${PIXELS[@]}")
  echo "[smoke] out_root=$OUT_ROOT eval dtype=$DTYPE" >&2
fi

for arm in $ARMS; do
  # shellcheck disable=SC2046
  python "$DIR/train_ft.py" --model_path "$MODEL_PATH" --train_data "$TRAIN_DATA" --image_root "$IMAGE_ROOT" \
    --output_dir "$OUT_ROOT/ckpt_$arm" $(arm_flags "$arm") "${SMOKE_TRAIN[@]+"${SMOKE_TRAIN[@]}"}" \
    "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"
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
[[ "$SMOKE" == 1 ]] && echo "[smoke] all stages ran; scores above are not meaningful" >&2
exit 0
