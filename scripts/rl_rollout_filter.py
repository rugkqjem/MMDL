"""Difficulty filtering for GRPO: sample each candidate question n times with the base model, record the pass rate.

GRPO only learns from groups with both right and wrong answers, so questions solved 8/8 or 0/8 carry no signal.
Inputs, sampling and scoring are the eval's own code, not re-implementations:
  - infer.prepare_sample()   prompt, image min/max pixels, process_vision_info, chat template
  - infer.make_llm() / make_sampling(n)   vLLM engine, VL sampling recipe, model revision, dtype, seed
  - resample_wrong.is_correct()   extract_final_answer() + score_qwen(), the primary metric
Any infer.py argument passes through (--model_path, --max_new_tokens, ...), as in resample_wrong.py.

One deliberate difference from the eval: a truncated sample (finish_reason == 'length') counts as wrong in
`correct`, because the RL reward will penalize truncation. The eval's verdict is kept as `correct_eval`.
--max_new_tokens defaults to 8192 here (eval: 16384): set it to the RL max response length.

Two-stage sampling: draw --stage1_n first; if all are correct the question is too easy and is not sampled further.
Resumable: re-running with the same --out skips ids already written.

  # pilot: measure speed, then read pilot_resp.jsonl to check the grading by eye
  python scripts/rl_rollout_filter.py --data outputs/decontam/virl/clean.jsonl --image_root data/virl39k_raw \\
      --out outputs/rl_rollout/pilot.jsonl --limit 200 --shuffle --save_responses outputs/rl_rollout/pilot_resp.jsonl
  # full run
  nohup python scripts/rl_rollout_filter.py --data outputs/decontam/virl/clean.jsonl --image_root data/virl39k_raw \\
      --out outputs/rl_rollout/rollouts.jsonl > rl_rollout.log 2>&1 &
  # pipeline test without GPU / vLLM / model download
  python scripts/rl_rollout_filter.py --data data/virl39k/sub.jsonl --image_root data/virl39k_raw \\
      --out /tmp/t.jsonl --limit 50 --dry_run
"""
import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer  # noqa: E402
from resample_wrong import is_correct  # noqa: E402


def to_sample(row, image_root):
    """Candidate row -> the HF-MMMU-shaped dict infer.prepare_sample() expects."""
    from PIL import Image
    paths = row['image'] if isinstance(row['image'], list) else [row['image']]
    imgs = [Image.open(p if os.path.isabs(p) else os.path.join(image_root, p)).convert('RGB') for p in paths]
    return {'id': row['id'], 'question': row['question'], 'question_type': row['question_type'],
            'options': repr(row['options']) if row['options'] else '', 'answer': row['answer'],
            **{f'image_{i + 1}': im for i, im in enumerate(imgs)}}


class VLLMEngine:
    def __init__(self, args):
        from transformers import AutoProcessor
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.model_path, revision=args.model_revision or None)
        self.llm = infer.make_llm(args)

    def build(self, row, image_root):
        return infer.prepare_sample(to_sample(row, image_root), row.get('category', ''), self.processor, self.args)

    def generate(self, inputs, n, seed):
        a = copy.copy(self.args)
        a.seed = seed
        outs = self.llm.generate(inputs, sampling_params=infer.make_sampling(a, n=n), use_tqdm=False)
        return [[(o.text, o.finish_reason, len(o.token_ids)) for o in r.outputs] for r in outs]


class DryRunEngine:
    """Fake generations: tests data loading, prompt parity, scoring and bookkeeping without a GPU."""

    def __init__(self, args):
        self.args = args

    def build(self, row, image_root):
        sample = to_sample(row, image_root)
        _, prompt, options = infer.build_messages(sample, self.args.min_pixels, self.args.max_pixels)
        return row, {'id': row['id'], 'options': options, 'answer': row['answer'], 'prompt': prompt}

    def generate(self, inputs, n, seed):
        res = []
        for row in inputs:
            rng, p = random.Random(f"{row['id']}-{seed}"), random.Random(row['id']).random()
            group = []
            for _ in range(n):
                u = rng.random()
                if u < 0.1:
                    group.append(('Let me re-check the figure. ' * 40 + 'Answer: A', 'length', self.args.max_new_tokens))
                    continue
                wrong = 'A' if row['answer'] != 'A' else 'B'
                ans = row['answer'] if u < 0.1 + 0.9 * p else wrong
                group.append((f'Step by step...\nThe answer is **({ans})**.', 'stop', 200))
            res.append(group)
        return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', required=True, help='rl_prepare_virl39k.py output (after decontaminate.py)')
    p.add_argument('--image_root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--stage1_n', type=int, default=4, help='0 = draw all n at once')
    p.add_argument('--chunk', type=int, default=256, help='questions per generate() call')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--shuffle', action='store_true', help='shuffle before --limit (random pilot sample)')
    p.add_argument('--save_responses', default=None, help='also write every response here (large)')
    p.add_argument('--dry_run', action='store_true')
    own, rest = p.parse_known_args()
    if '--max_new_tokens' not in rest:
        rest += ['--max_new_tokens', '8192']
    args = infer.parse_args(rest + ['--output', own.out])
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

    with open(own.data, encoding='utf-8') as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if own.shuffle:
        random.Random(args.seed).shuffle(rows)
    if own.limit:
        rows = rows[:own.limit]
    done = set()
    if os.path.exists(own.out):
        with open(own.out, encoding='utf-8') as f:
            done = {json.loads(l)['id'] for l in f if l.strip()}
    todo = [r for r in rows if r['id'] not in done]
    print(f'[rollout] {len(rows)} questions, {len(done)} done, {len(todo)} to go; '
          f'n={own.n} stage1_n={own.stage1_n} max_new_tokens={args.max_new_tokens}', file=sys.stderr)
    if not todo:
        return
    os.makedirs(os.path.dirname(os.path.abspath(own.out)), exist_ok=True)
    engine = DryRunEngine(args) if own.dry_run else VLLMEngine(args)
    s1 = own.stage1_n if 0 < own.stage1_n < own.n else own.n

    fout = open(own.out, 'a', encoding='utf-8')
    fresp = open(own.save_responses, 'a', encoding='utf-8') if own.save_responses else None
    t0, tok_total, n_done, n_err = time.time(), 0, 0, 0
    for c in range(0, len(todo), own.chunk):
        items, inputs, recs = [], [], []
        for r in todo[c:c + own.chunk]:
            try:
                inp, rec = engine.build(r, own.image_root)
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f'[skip] {r["id"]}: {e}', file=sys.stderr)
                continue
            human = next(x['value'] for x in r['conversations'] if x['from'] == 'human')
            assert rec['prompt'] == human, f'{r["id"]}: prompt differs from the eval prompt builder'
            items.append(r); inputs.append(inp); recs.append(rec)
        if not items:
            continue
        samples = [[] for _ in items]

        def run(idx, n, seed):
            nonlocal tok_total
            for i, group in zip(idx, engine.generate([inputs[i] for i in idx], n, seed)):
                for text, finish, ntok in group:
                    ok_eval = bool(is_correct(recs[i], text))
                    samples[i].append({'response': text, 'finish_reason': finish, 'num_output_tokens': ntok,
                                       'correct_eval': ok_eval, 'correct': ok_eval and finish != 'length'})
                    tok_total += ntok

        run(list(range(len(items))), s1, args.seed)
        early = set()
        if s1 < own.n:
            rest_idx = [i for i in range(len(items)) if not all(s['correct'] for s in samples[i])]
            early = set(range(len(items))) - set(rest_idx)
            if rest_idx:
                run(rest_idx, own.n - s1, args.seed + 1)

        for i, r in enumerate(items):
            ss = samples[i]
            fout.write(json.dumps({
                'id': r['id'], 'category': r.get('category', ''), 'question_type': r['question_type'],
                'n': len(ss), 'n_correct': sum(s['correct'] for s in ss),
                'n_correct_eval': sum(s['correct_eval'] for s in ss),
                'n_truncated': sum(s['finish_reason'] == 'length' for s in ss),
                'mean_output_tokens': sum(s['num_output_tokens'] for s in ss) / len(ss),
                'early_stop': i in early}, ensure_ascii=False) + '\n')
            if fresp:
                fresp.write(json.dumps({'id': r['id'], 'answer': r['answer'], 'options': r['options'],
                                        'samples': ss}, ensure_ascii=False) + '\n')
        fout.flush()
        if fresp:
            fresp.flush()
        n_done += len(items)
        el = time.time() - t0
        eta = (len(todo) - n_done) * el / n_done
        print(f'[rollout] {n_done}/{len(todo)}  {tok_total / el:,.0f} tok/s  {n_done / el * 3600:,.0f} q/h  '
              f'ETA {eta / 3600:.1f} h  (too easy after stage 1 in this chunk: {len(early)})', file=sys.stderr)
    fout.close()
    if fresp:
        fresp.close()
    print(f'[rollout] done: {n_done} written, {n_err} skipped -> {own.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
