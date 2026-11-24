#!/usr/bin/env bash
set -euo pipefail
cd /workspace/MMDL
. /workspace/venv/bin/activate
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0  # xet chunk cache held ~7GB during the first run and filled the 32GB disk
ts() { date "+%H:%M:%S"; }
echo "[$(ts)] SMOKE start"; df -h / | tail -1
[[ -f data/smoke/train.jsonl ]] || python scripts/prepare_mmmu_sft.py --subjects Math --output_dir data/smoke
rm -rf outputs/ablation_smoke
bash scripts/run_ablation.sh --smoke --delete_ckpt_after_eval --train_data data/smoke/train.jsonl --image_root data/smoke/images
echo "[$(ts)] SMOKE DONE"; df -h / | tail -1
cat outputs/ablation_smoke/comparison.md | head -8
echo "[$(ts)] RESAMPLE start"
python scripts/resample_wrong.py --subjects Agriculture Geography Sociology Chemistry --k 8 --output_dir outputs/resample_wrong
echo "[$(ts)] RESAMPLE DONE"
