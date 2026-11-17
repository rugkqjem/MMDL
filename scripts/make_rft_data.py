"""RFT (rejection-sampling fine-tuning) data from the existing rollouts: no new generation.

For each final-train question, keep rollout responses that
  - ended on their own (finish_reason == 'stop', not cut at 8192 tokens),
  - are graded correct by the eval/reward grader (rl_reward.score_response),
then take up to --per_q of them per question (random, seeded). Questions in the exclude lists are skipped.
Optional extra filters (off by default, team decision 2026-11-17): --max_tokens, and --max_loop_share
(repetition/hedging character share from loop_onset.unit_flags + loop_share; 0.4 drops ~9% of the responses).

Output is the qwen-vl-finetune format scripts/train_ft.py reads (no <image> tag -> images go before the text,
the eval layout), plus bookkeeping keys train_ft.py ignores:
    {"image": [...], "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": response}],
     "id": ..., "sample": k, "tokens": n}

    python scripts/make_rft_data.py                       # -> data/rft/rft_train.jsonl + rft_train.md
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpo_pairs import read_ids, read_jsonl  # noqa: E402
from loop_onset import loop_share, unit_flags  # noqa: E402
from rl_reward import score_response  # noqa: E402


def eligible(q, samples, max_tokens, max_loop_share):
    """Correct, self-terminated, not too long, not loop-like. Returns (kept samples, reject reasons)."""
    keep, why = [], Counter()
    for s in samples:
        if s['finish_reason'] != 'stop':
            why['truncated'] += 1
        elif not score_response(s['response'], q['answer'], q['options'], truncated=False)['correct']:
            why['wrong'] += 1
        elif s['num_output_tokens'] > max_tokens:
            why['too long'] += 1
        elif max_loop_share < 1 and loop_share(unit_flags(s['response']), 0, len(s['response'])) > max_loop_share:
            why['loop-like'] += 1
        else:
            keep.append(s)
    return keep, why


def to_row(q, s):
    imgs = q['images'] if isinstance(q['images'], list) else [q['images']]
    return {'image': imgs,
            'conversations': [{'from': 'human', 'value': q['prompt']}, {'from': 'gpt', 'value': s['response']}],
            'id': q['id'], 'sample': s['_k'], 'tokens': s['num_output_tokens'],
            'category': q.get('category', '')}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='data/rl/train_final.jsonl')
    p.add_argument('--resp', nargs='+', default=['outputs/rl_rollout/rollouts_resp.jsonl',
                                                 'outputs/rl_rollout/pilot_resp.jsonl'])
    p.add_argument('--exclude', nargs='*', default=['data/rl/exclude_format_only.txt',
                                                    'data/rl/exclude_label_issues.txt'])
    p.add_argument('--per_q', type=int, default=2, help='responses per question at most')
    p.add_argument('--max_tokens', type=int, default=8192, help='skip correct responses longer than this (8192 = off)')
    p.add_argument('--max_loop_share', type=float, default=1.0,
                   help='skip responses whose repetition/hedging text share is above this (1 = off)')
    p.add_argument('--out', default='data/rft/rft_train.jsonl')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)

    qs = {r['id']: r for r in read_jsonl(a.train)}
    excluded = set().union(*(read_ids(f) for f in a.exclude if os.path.exists(f)))
    samples, seen = defaultdict(list), set()
    for path in a.resp:
        for g in read_jsonl(path):
            if g['id'] not in qs or g['id'] in seen:
                continue
            seen.add(g['id'])
            for k, s in enumerate(g['samples']):
                samples[g['id']].append(dict(s, _k=k))

    rows, why, n_q_none = [], Counter(), 0
    for qid in sorted(samples):
        if qid in excluded:
            why['excluded question'] += 1
            continue
        keep, w = eligible(qs[qid], samples[qid], a.max_tokens, a.max_loop_share)
        why.update(w)
        if not keep:
            n_q_none += 1
            continue
        for s in rng.sample(keep, min(a.per_q, len(keep))):
            rows.append(to_row(qs[qid], s))
    rng.shuffle(rows)

    assert rows, 'no RFT rows'
    assert all(r['id'] in qs and r['id'] not in excluded for r in rows)
    assert all(c <= a.per_q for c in Counter(r['id'] for r in rows).values())
    assert all('<|im_end|>' not in r['conversations'][1]['value'] for r in rows)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    toks = sorted(r['tokens'] for r in rows)
    n_q = len({r['id'] for r in rows})
    lines = [f'# RFT data ({len(rows)} responses / {n_q} questions)', '',
             f'- args: `{json.dumps(vars(a), ensure_ascii=False)}`',
             f'- questions with rollouts: {len(samples)}, no usable correct response: {n_q_none}',
             f'- response tokens: median {median(toks)}, p90 {toks[int(0.9 * (len(toks) - 1))]}, max {toks[-1]}',
             f'- skipped responses: {dict(why)}',
             f'- by category: {dict(Counter(r["category"] for r in rows).most_common())}']
    with open(os.path.splitext(a.out)[0] + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
