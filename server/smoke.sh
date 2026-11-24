#!/usr/bin/env bash
# vast.ai smoke driver: same stages as `run_ablation.sh --smoke`, one arm at a time so the 32GB disk holds
# at most one 9GB checkpoint (train -> eval -> delete).
set -euo pipefail
cd /workspace/MMDL
. /workspace/venv/bin/activate
OUT=outputs/ablation_smoke
ts() { date '+%H:%M:%S'; }
echo "[$(ts)] prepare data"
python scripts/prepare_mmmu_sft.py --subjects Math --output_dir data/smoke
COMMON=(--smoke --train_data data/smoke/train.jsonl --image_root data/smoke/images --out_root "$OUT")
first=1
for arm in deepstack lora both; do
  echo "[$(ts)] arm $arm"
  extra=(); [[ $first == 1 ]] || extra=(--skip_baseline)
  bash scripts/run_ablation.sh "${COMMON[@]}" --arms "$arm" "${extra[@]+"${extra[@]}"}"
  rm -rf "$OUT/ckpt_$arm"; first=0
  df -h / | tail -1
done
echo "[$(ts)] compare"
python scripts/compare_ablation.py --runs baseline=$OUT/eval_baseline deepstack=$OUT/eval_deepstack \
  lora=$OUT/eval_lora both=$OUT/eval_both --output $OUT/comparison.md
echo "[$(ts)] SMOKE DONE"
