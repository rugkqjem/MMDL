"""Check rl_reward.py against responses that were already scored. CPU only.

1) --mmmu: the baseline MMMU-val run. The reward's correct_eval must equal scores_qwen.csv `correct` on every
   multiple-choice question (the RL reward and the reported metric are the same scorer).
2) --rollouts_resp: rl_rollout_filter.py --save_responses output. The reward must reproduce each sample's
   `correct_eval` and `correct`, and the script reports, per truncation penalty, how many groups would carry no
   learning signal (all rewards equal), the offline counterpart of TRL's frac_reward_zero_std. These rollouts used
   the eval sampling recipe, so compare the pilot's frac_reward_zero_std with this number, not expect equality.

  python scripts/check_rl_reward.py --mmmu
  python scripts/check_rl_reward.py --rollouts_resp outputs/rl_rollout/rollouts_resp.jsonl \\
      --ids data/rl/train_noA.jsonl --penalties 0 0.25 0.5 1
Exits non-zero on any mismatch.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_reward import score_response  # noqa: E402

BASE = 'outputs/qwen3vl4b_mmmu_val'


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def check_mmmu(pred_path, scores_path, penalty):
    preds = read_jsonl(pred_path)
    with open(scores_path, encoding='utf-8') as f:
        scores = {r['id']: r for r in csv.DictReader(f)}
    mc = [p for p in preds if p['options']]
    bad, rewards = [], Counter()
    for p in mc:
        r = score_response(p['response'], p['answer'], p['options'], p['finish_reason'] == 'length', penalty)
        if r['correct_eval'] != (scores[p['id']]['correct'] == 'True'):
            bad.append(p['id'])
        rewards[r['reward']] += 1
    print(f'[mmmu] multiple-choice {len(mc)} of {len(preds)} (open questions are never used as RL rewards)')
    print(f'[mmmu] correct_eval == scores_qwen.csv correct: {len(mc) - len(bad)}/{len(mc)}')
    print(f'[mmmu] reward distribution at penalty {penalty}: {dict(sorted(rewards.items()))}')
    if bad:
        print(f'[mmmu] MISMATCH: {bad[:10]}')
    return not bad


def check_rollouts(path, ids_path, penalties):
    groups = read_jsonl(path)
    if ids_path:
        keep = {r['id'] for r in read_jsonl(ids_path)}
        groups = [g for g in groups if g['id'] in keep]
    n_s, bad = 0, []
    scored = []  # per group: list of (correct, truncated)
    for g in groups:
        row = []
        for s in g['samples']:
            r = score_response(s['response'], g['answer'], g['options'], s['finish_reason'] == 'length')
            n_s += 1
            if r['correct_eval'] != s['correct_eval'] or r['correct'] != s['correct']:
                bad.append(g['id'])
            row.append((r['correct'], r['truncated']))
        scored.append(row)
    print(f'[rollouts] {len(groups)} questions, {n_s} samples; correct/correct_eval reproduced: '
          f'{n_s - len(bad)}/{n_s}')
    if bad:
        print(f'[rollouts] MISMATCH in: {sorted(set(bad))[:10]}')
    print('\n| penalty | mean reward | groups with no signal (all rewards equal) |\n|---:|---:|---:|')
    for p in penalties:
        vals = [[1.0 if c else (-p if t else 0.0) for c, t in row] for row in scored]
        zero = sum(len(set(v)) == 1 for v in vals)
        mean = sum(map(sum, vals)) / max(sum(map(len, vals)), 1)
        print(f'| {p} | {mean:.3f} | {zero} ({zero / max(len(vals), 1):.1%}) |')
    return not bad


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--mmmu', action='store_true')
    p.add_argument('--pred', default=f'{BASE}/predictions.jsonl')
    p.add_argument('--scores', default=f'{BASE}/scores_qwen.csv')
    p.add_argument('--rollouts_resp', default=None)
    p.add_argument('--ids', default=None, help='restrict to the questions in this JSONL (e.g. train_noA.jsonl)')
    p.add_argument('--penalties', type=float, nargs='+', default=[0.0, 0.25, 0.5, 1.0])
    a = p.parse_args()
    if not (a.mmmu or a.rollouts_resp):
        p.error('give --mmmu and/or --rollouts_resp')
    ok = True
    if a.mmmu:
        ok &= check_mmmu(a.pred, a.scores, a.penalties[-1] if len(a.penalties) == 1 else 0.5)
    if a.rollouts_resp:
        ok &= check_rollouts(a.rollouts_resp, a.ids, a.penalties)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
