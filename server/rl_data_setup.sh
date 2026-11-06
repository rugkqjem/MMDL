#!/usr/bin/env bash
# vast.ai (A100 80GB, disk >= 150GB): fresh instance -> GRPO data ready for the full rollout.
#   env (uv venv, Python 3.12, requirements.lock.txt) -> model + ViRL39K download -> rl_prepare_virl39k.py
#   -> decontaminate.py -> 200-question pilot rollout -> pilot bucket report
# Stops after the pilot on purpose: read the pilot responses before paying for the full rollout.
#
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/rugkqjem/MMDL/rl-grpo/server/rl_data_setup.sh)"
#   # or, once the repo is cloned:
#   nohup bash server/rl_data_setup.sh > /workspace/rl_data_setup.log 2>&1 &   tail -f /workspace/rl_data_setup.log
#
# Re-running is safe: finished steps are skipped (venv, downloads, prepared JSONL, decontam output, pilot ids).
# Knobs (env vars): BRANCH, EXCLUDE, MAX_PER_CAT, PILOT_N, MAX_NEW_TOKENS, NO_EMBED=1 (skip CLIP in decontam)
set -euo pipefail

WS=/workspace
REPO=$WS/MMDL
VENV=$WS/venv
BRANCH=${BRANCH:-rl-grpo}
EXCLUDE=${EXCLUDE:-GradeSchool}
MAX_PER_CAT=${MAX_PER_CAT:-3000}
PILOT_N=${PILOT_N:-200}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}   # = planned GRPO max response length
MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17  # infer.py MODEL_REVISION

export HF_HOME=$WS/hf_home               # model + CLIP cache on the big disk
export HF_DATA=$WS/hf_data               # MMMU / MMMU-Pro datasets cache (--data_root, reused by later evals)
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0   # xet chunk cache held ~7GB in the first run and filled the disk
export VLLM_WORKER_MULTIPROC_METHOD=spawn

ts() { date "+%H:%M:%S"; }
step() { echo; echo "[$(ts)] ===== $* ====="; df -h $WS | tail -1; }

step "0. GPU / disk check"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
avail_gb=$(df -BG --output=avail $WS | tail -1 | tr -dc 0-9)
if (( avail_gb < 100 )); then
  echo "only ${avail_gb}GB free on $WS (need ~100GB+: env 15, model 9, ViRL39K, MMMU+Pro, later checkpoints)"; exit 1
fi

step "1. repo ($BRANCH)"
if [[ -d $REPO/.git ]]; then
  git -C $REPO fetch -q origin && git -C $REPO checkout -q $BRANCH && git -C $REPO pull -q --ff-only
else
  git clone -q -b $BRANCH https://github.com/rugkqjem/MMDL.git $REPO
fi
cd $REPO
git log --oneline -1
# transformers 5.x CLIP fix must be in (pushed from the Mac); without it decontaminate.py crashes at the image step
grep -q 'getattr(out, "pooler_output", out)' scripts/decontaminate.py \
  || { echo "decontaminate.py is missing the CLIP fix: push the latest $BRANCH from the Mac first"; exit 1; }

step "2. venv (Python 3.12 + requirements.lock.txt)"
if [[ ! -x $VENV/bin/python ]]; then
  command -v uv >/dev/null || pip install -q uv
  uv venv -q --python 3.12 $VENV
  # torch / vllm wheels are large: uv is much faster than pip here
  VIRTUAL_ENV=$VENV uv pip install -q -r requirements.lock.txt
  VIRTUAL_ENV=$VENV uv pip install -q pandas pyarrow datasketch imagehash
fi
. $VENV/bin/activate
python -c "import vllm, transformers, torch; print('vllm', vllm.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

step "3. downloads (model + ViRL39K in parallel)"
hf download Qwen/Qwen3-VL-4B-Instruct --revision $MODEL_REVISION > /dev/null &
pid_model=$!
hf download TIGER-Lab/ViRL39K --repo-type dataset --local-dir data/virl39k_raw > /dev/null &
pid_virl=$!
wait $pid_model && echo "[$(ts)] model ok"
wait $pid_virl && echo "[$(ts)] ViRL39K ok"
du -sh data/virl39k_raw

step "4. ViRL39K -> JSONL (exclude: $EXCLUDE, max per category: $MAX_PER_CAT)"
mkdir -p data/virl39k
[[ -s data/virl39k/all_stats.txt ]] || python scripts/rl_prepare_virl39k.py --src data/virl39k_raw \
  --out data/virl39k/all.jsonl | tee data/virl39k/all_stats.txt
if [[ ! -s data/virl39k/sub.jsonl ]]; then
  python scripts/rl_prepare_virl39k.py --src data/virl39k_raw --out data/virl39k/sub.jsonl \
    --exclude_categories $EXCLUDE --max_per_category $MAX_PER_CAT
fi
wc -l data/virl39k/sub.jsonl

step "5. MMMU decontamination (first run downloads MMMU dev/val/test + MMMU-Pro and builds the reference)"
DECON=outputs/decontam/virl
if [[ ! -s $DECON/clean.jsonl ]]; then
  python scripts/decontaminate.py --data data/virl39k/sub.jsonl --image_root data/virl39k_raw \
    --out $DECON --data_root $HF_DATA ${NO_EMBED:+--no_embed}
fi
# default decision is 'remove' for every flagged row, so clean.jsonl is already the conservative set.
# Reviewing review.html later can only add rows back (decisions.csv: remove -> keep, then --apply).
cat $DECON/report.md
wc -l $DECON/clean.jsonl

step "6. pilot rollout ($PILOT_N questions x 8 samples, max_new_tokens $MAX_NEW_TOKENS)"
mkdir -p outputs/rl_rollout
python scripts/rl_rollout_filter.py --data $DECON/clean.jsonl --image_root data/virl39k_raw \
  --out outputs/rl_rollout/pilot.jsonl --limit $PILOT_N --shuffle \
  --save_responses outputs/rl_rollout/pilot_resp.jsonl \
  --data_root $HF_DATA --max_new_tokens $MAX_NEW_TOKENS

step "7. pilot report"
python scripts/rl_select.py --rollouts outputs/rl_rollout/pilot.jsonl --data $DECON/clean.jsonl \
  --out outputs/rl_rollout/pilot_select/train.jsonl --keep_trunc_mixed
python - <<'EOF'
import json
rows = [json.loads(l) for l in open('outputs/rl_rollout/pilot_resp.jsonl')]
s = [x for r in rows for x in r['samples']]
fin = [x for x in s if x['finish_reason'] != 'length']
marked = sum(('answer is' in x['response'].lower()) or ('answer:' in x['response'].lower()) or ('\\boxed' in x['response'])
             for x in fin)
print(f"\nsamples {len(s)} | truncated {len(s) - len(fin)} ({(len(s) - len(fin)) / len(s):.1%}) | "
      f"finished with an answer marker {marked}/{len(fin)} ({marked / max(len(fin), 1):.1%})")
print("-> if the marker rate is low, the parser falls back to the full response: read pilot_resp.jsonl before going on")
EOF
echo
echo "[$(ts)] DONE. Next:"
echo "  1) read outputs/rl_rollout/pilot_resp.jsonl (samples with correct_eval=false): is the grading right?"
echo "  2) full-run time ~= $(wc -l < $DECON/clean.jsonl) questions / (q/h from the step-6 log)"
echo "  3) nohup python scripts/rl_rollout_filter.py --data $DECON/clean.jsonl --image_root data/virl39k_raw \\"
echo "       --out outputs/rl_rollout/rollouts.jsonl --data_root $HF_DATA --max_new_tokens $MAX_NEW_TOKENS > rl_rollout.log 2>&1 &"
