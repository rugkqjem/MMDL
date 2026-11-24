"""Pick the GRPO training set from rl_rollout_filter.py results and write a report.

Keep rule (default): 1 <= n_correct <= n-1 -> the group mixes right and wrong answers.
--keep_trunc_mixed also keeps 0-correct questions where only SOME samples were truncated: with a truncation
penalty in the reward (truncated < finished-but-wrong) those groups still have reward variance, and they are the
repetition-loop cases we want to fix.

Output JSONL, one question per row:
  problem   '<image>' x num_images + the eval prompt (EasyR1 splits on '<image>' to place images; images go
            before the text, as in infer.build_messages)
  prompt    the eval prompt without image tags (for TRL or custom loops)
  images    image paths (absolute if --image_root is given)
  answer, options, question_type   what score.score_qwen() needs to compute the reward
  category, pass_rate, trunc_rate

  python scripts/rl_select.py --rollouts outputs/rl_rollout/rollouts.jsonl --data outputs/decontam/virl/clean.jsonl \\
      --out data/rl/train.jsonl --image_root data/virl39k_raw --keep_trunc_mixed --val_size 300
Writes train.jsonl, train_val.jsonl (if --val_size), train_report.md.
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict


def bucket(r):
    n, c, t = r['n'], r['n_correct'], r['n_truncated']
    if r.get('early_stop') or c == n:
        return 'too_easy'
    if c > 0:
        return 'mixed'
    if 0 < t < n:
        return 'zero_trunc_mixed'
    return 'zero_all_trunc' if t == n else 'zero'


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--rollouts', required=True)
    p.add_argument('--data', required=True, help='the JSONL the rollouts were run on')
    p.add_argument('--out', required=True)
    p.add_argument('--image_root', default=None, help='if set, image paths are written as absolute paths')
    p.add_argument('--keep_trunc_mixed', action='store_true')
    p.add_argument('--min_options', type=int, default=3, help='same as rl_rollout_filter.py (0 = keep all)')
    p.add_argument('--max_items', type=int, default=None, help='cap, stratified by category')
    p.add_argument('--val_size', type=int, default=0, help='held-out RL validation split')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    data = {r['id']: r for r in read_jsonl(args.data) if len(r.get('options') or []) >= args.min_options}
    ro = [r for r in read_jsonl(args.rollouts) if r['id'] in data]
    keep_b = {'mixed'} | ({'zero_trunc_mixed'} if args.keep_trunc_mixed else set())
    kept = [r for r in ro if bucket(r) in keep_b]
    rng = random.Random(args.seed)
    if args.max_items and len(kept) > args.max_items:
        by = defaultdict(list)
        for r in kept:
            by[r['category']].append(r)
        total, kept = len(kept), []
        for _, v in sorted(by.items()):
            rng.shuffle(v)
            kept += v[:max(1, round(args.max_items * len(v) / total))]

    out = []
    for r in kept:
        d = data[r['id']]
        prompt = next(c['value'] for c in d['conversations'] if c['from'] == 'human')
        imgs = d['image'] if isinstance(d['image'], list) else [d['image']]
        if args.image_root:
            imgs = [i if os.path.isabs(i) else os.path.abspath(os.path.join(args.image_root, i)) for i in imgs]
        out.append({'id': r['id'], 'problem': '<image>' * len(imgs) + prompt, 'prompt': prompt, 'images': imgs,
                    'answer': d['answer'], 'options': d['options'], 'question_type': d['question_type'],
                    'category': d.get('category', ''), 'pass_rate': r['n_correct'] / r['n'],
                    'trunc_rate': r['n_truncated'] / r['n']})
    rng.shuffle(out)
    val, train = out[:args.val_size], out[args.val_size:]
    stem = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for path, rows in [(args.out, train)] + ([(stem + '_val.jsonl', val)] if val else []):
        with open(path, 'w', encoding='utf-8') as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')

    # ---------------------------------------------------------------- report
    b, n_tot = Counter(bucket(r) for r in ro), len(ro)
    hist = Counter(r['n_correct'] for r in ro if not r.get('early_stop'))
    lines = [f'# GRPO data selection: `{args.rollouts}`', '',
             f'- questions with rollouts: **{n_tot}**',
             f'- kept: **{len(out)}** (train {len(train)}, val {len(val)}); rule: {sorted(keep_b)}', '',
             '| bucket | count | share |', '|---|---:|---:|']
    for k in ['too_easy', 'mixed', 'zero_trunc_mixed', 'zero_all_trunc', 'zero']:
        lines.append(f'| {k} | {b.get(k, 0)} | {b.get(k, 0) / max(n_tot, 1):.1%} |')
    lines += ['', 'too_easy includes questions stopped after stage 1 (all correct). '
              '`correct` counts truncated samples as wrong; the eval verdict is in rollouts.jsonl `n_correct_eval`.',
              '', '## n_correct histogram (fully sampled questions)', '', '| n_correct | count |', '|---:|---:|']
    lines += [f'| {k} | {hist[k]} |' for k in sorted(hist)]
    lines += ['', '## By category', '',
              '| category | total | kept | too easy | zero | mean pass | mean trunc |',
              '|---|---:|---:|---:|---:|---:|---:|']
    by_c, kept_ids = defaultdict(list), {x['id'] for x in out}
    for r in ro:
        by_c[r['category']].append(r)
    for c, v in sorted(by_c.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {c} | {len(v)} | {sum(r['id'] in kept_ids for r in v)} | "
                     f"{sum(bucket(r) == 'too_easy' for r in v)} | {sum(bucket(r).startswith('zero') for r in v)} | "
                     f"{sum(r['n_correct'] / r['n'] for r in v) / len(v):.2f} | "
                     f"{sum(r['n_truncated'] / r['n'] for r in v) / len(v):.2f} |")
    tr = sum(r['n_truncated'] for r in ro) / max(sum(r['n'] for r in ro), 1)
    lines += ['', f'Truncation rate over all samples: **{tr:.1%}**']
    with open(stem + '_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"\n-> {args.out}{', ' + stem + '_val.jsonl' if val else ''}, {stem}_report.md")


if __name__ == '__main__':
    main()
