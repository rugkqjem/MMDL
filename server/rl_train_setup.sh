#!/usr/bin/env bash
# Lab A100, >= 60GB free disk: env setup -> GRPO pilot (20 optimizer steps).
#   eval env ($WS/venv, requirements.lock.txt) + TRL/PEFT installed with the lock as a constraint
#   -> model + ViRL39K images -> CPU checks -> pilot -> pilot summary
# Stops after the pilot on purpose: read the summary (signal, truncation, time per step) before the full run.
#
# Before running, copy the GRPO data from the Mac (data/ is gitignored):
#   scp -r ~/Desktop/MMMU/MMDL/data/rl <user>@<host>:<WS>/MMDL/data/
#   WS=<WS> nohup bash server/rl_train_setup.sh > rl_train_setup.log 2>&1 &   tail -f rl_train_setup.log
# Re-running is safe: finished steps are skipped. Knobs (env vars): WS (work dir, default /workspace), BRANCH, PILOT_STEPS, SAMPLING (train|eval),
#   TRUNC_PENALTY, PILOT_NAME (output dir under outputs/grpo, default pilot_$SAMPLING; must not hold an earlier pilot),
#   EXTRA (more train_grpo.py args, e.g. "--per_device_train_batch_size 1 --gradient_accumulation_steps 32")
set -euo pipefail

WS=${WS:-/workspace}
REPO=$WS/MMDL
VENV=$WS/venv
BRANCH=${BRANCH:-rl-grpo}
PILOT_STEPS=${PILOT_STEPS:-20}
SAMPLING=${SAMPLING:-train}
TRUNC_PENALTY=${TRUNC_PENALTY:-0.5}
EXTRA=${EXTRA:-}
PILOT_NAME=${PILOT_NAME:-pilot_${SAMPLING}}
MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17  # infer.py MODEL_REVISION

export HF_HOME=$WS/hf_home
export HF_DATA=$WS/hf_data
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ts() { date "+%H:%M:%S"; }
step() { echo; echo "[$(ts)] ===== $* ====="; df -h $WS | tail -1; }

step "0. GPU / disk check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
avail_gb=$(df -BG --output=avail $WS | tail -1 | tr -dc 0-9)
(( avail_gb >= 60 )) || { echo "only ${avail_gb}GB free on $WS"; exit 1; }

step "1. repo ($BRANCH)"
if [[ -d $REPO/.git ]]; then
  git -C $REPO fetch -q origin && git -C $REPO checkout -q $BRANCH && git -C $REPO pull -q --ff-only
else
  git clone -q -b $BRANCH https://github.com/rugkqjem/MMDL.git $REPO
fi
cd $REPO
git log --oneline -1
[[ -s data/rl/train_final.jsonl ]] || { echo "data/rl/train_final.jsonl missing: scp data/rl from the Mac (see header)"; exit 1; }

step "2. env: eval lock + TRL/PEFT (eval versions pinned by -c requirements.lock.txt)"
if [[ ! -x $VENV/bin/python ]]; then
  command -v uv >/dev/null || pip install -q uv
  uv venv -q --python 3.12 $VENV
  VIRTUAL_ENV=$VENV uv pip install -q -r requirements.lock.txt
fi
VIRTUAL_ENV=$VENV uv pip install -q -r requirements.rl.txt -c requirements.lock.txt
. $VENV/bin/activate
python -c "import vllm, transformers, torch, trl, peft; print('vllm', vllm.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__, '| trl', trl.__version__, '| peft', peft.__version__)"

step "3. downloads (model + ViRL39K images)"
hf download Qwen/Qwen3-VL-4B-Instruct --revision $MODEL_REVISION > /dev/null && echo "[$(ts)] model ok"
if [[ ! -d data/virl39k_raw/images ]]; then
  hf download TIGER-Lab/ViRL39K --repo-type dataset --local-dir data/virl39k_raw > /dev/null
  python -c "import sys; sys.path.insert(0, 'scripts'); import rl_prepare_virl39k as m; m.ensure_images('data/virl39k_raw')"
fi
python - <<'EOF'
import json, os
rows = [json.loads(l) for f in ('data/rl/train_final.jsonl', 'data/rl/train_val_final.jsonl') for l in open(f)]
miss = [p for r in rows for p in r['images'] if not os.path.exists(os.path.join('data/virl39k_raw', p))]
print(f'images: {sum(len(r["images"]) for r in rows) - len(miss)} found, {len(miss)} missing', miss[:3])
assert not miss
EOF

step "4. CPU checks (scorer, reward, configs, prompt/image parity with the real processor)"
python scripts/test_score.py
python scripts/test_rl_reward.py
python scripts/check_rl_reward.py --mmmu
python scripts/test_train_grpo.py
python scripts/test_vllm_prompt_fix.py
python scripts/train_grpo.py --check_only --sampling $SAMPLING --trunc_penalty $TRUNC_PENALTY $EXTRA

step "5. pilot GRPO ($PILOT_STEPS optimizer steps, sampling $SAMPLING, trunc_penalty $TRUNC_PENALTY)"
PILOT=outputs/grpo/$PILOT_NAME
# metrics.jsonl is appended to: an earlier pilot in the same dir would be mixed into the summary
[[ ! -e $PILOT/metrics.jsonl ]] || { echo "$PILOT already has a pilot: set PILOT_NAME (e.g. pilot2_${SAMPLING})"; exit 1; }
python scripts/train_grpo.py --output_dir $PILOT --max_steps $PILOT_STEPS --save_steps 1000 \
  --sampling $SAMPLING --trunc_penalty $TRUNC_PENALTY $EXTRA

step "6. pilot summary"
python - "$PILOT/metrics.jsonl" <<'EOF'
import json, sys
allrows = list(map(json.loads, open(sys.argv[1])))
rows = [r for r in allrows if 'reward' in r]
def avg(k, rs=None): v = [r[k] for r in (rs or rows) if k in r]; return sum(v) / len(v) if v else float('nan')
print(f"steps logged {len(rows)} | reward {avg('reward'):.3f} | frac_reward_zero_std {avg('frac_reward_zero_std'):.3f} "
      f"(offline baseline at eval sampling: 0.000 with any penalty > 0)")
print(f"clipped_ratio {avg('completions/clipped_ratio'):.3f} (eval sampling rollouts: 0.168) | "
      f"mean length {avg('completions/mean_length'):.0f} | step_time {avg('step_time', allrows):.0f}s (all steps) | "
      f"peak mem {max(r.get('max_mem_gb', 0) for r in allrows):.1f} GB")
print('step  clipped  reward_truncated  IS_ratio  logp_diff  (clipped must equal reward_truncated; IS_ratio ~1)')
for r in rows:
    print(f"{r['step']:>4}  {r.get('completions/clipped_ratio', float('nan')):7.3f}  "
          f"{r.get('reward_parts/truncated', float('nan')):16.3f}  "
          f"{r.get('sampling/importance_sampling_ratio/mean', float('nan')):8.3f}  "
          f"{r.get('sampling/sampling_logp_difference/mean', float('nan')):9.3f}")
print(f"first -> last reward: {rows[0]['reward']:.3f} -> {rows[-1]['reward']:.3f}")
EOF
echo
echo "[$(ts)] DONE. full-run estimate: 780 questions x 8 / 32 per step = 195 steps per epoch x step_time above."
echo "  full run: nohup python scripts/train_grpo.py --output_dir outputs/grpo/run1 --sampling $SAMPLING $EXTRA > grpo_run1.log 2>&1 &"
