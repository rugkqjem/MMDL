"""Re-solve the baseline's wrong answers k times: knowledge gap or unstable reasoning?

For each question the baseline got wrong (primary metric: final-answer extraction + Qwen parser, as score.py), sample
k fresh responses with exactly the eval inputs and sampling recipe (infer.prepare_sample / make_sampling; k samples
per request via vLLM `n`), score each the same way, and count how many are correct.

- never solved (0/k): wrong every time -> the knowledge is missing; external data is the right lever.
- recovered (>=1/k): the model can get it, just not reliably -> reasoning is unstable; self-generated data
  (rejection sampling) or DPO is likely more effective than new knowledge.

    python scripts/resample_wrong.py --subjects Agriculture Geography Sociology Chemistry --k 8 \\
        [--pred outputs/qwen3vl4b_mmmu_val/predictions.jsonl] [--output_dir outputs/resample_wrong] [infer.py args...]
Any infer.py argument (--model_path, --data_root, --max_new_tokens, ...) passes through with its eval default.
Writes resample.jsonl (every response), summary.json and summary.md.
"""
import argparse
import json
import os
import time
from collections import Counter

import infer
from score import extract_final_answer, score_qwen


def is_correct(rec, response):
    text, _ = extract_final_answer(response, rec['options'])
    return score_qwen(dict(rec, response=response), text)[2]


def summarize(rows, k):
    by_subject = {}
    for r in rows:
        by_subject.setdefault(r['subject'], []).append(r)
    table = []
    for subject, rs in list(by_subject.items()) + [('ALL', rows)]:
        n = len(rs)
        never = sum(r['n_correct'] == 0 for r in rs)
        table.append({
            'subject': subject, 'wrong_at_baseline': n,
            'never_solved': never, 'recovered': n - never,
            'never_solved_pct': 100 * never / n if n else 0.0,
            'recovered_pct': 100 * (n - never) / n if n else 0.0,
            'mean_pass_rate_pct': 100 * sum(r['n_correct'] for r in rs) / (n * k) if n else 0.0,
            'n_correct_hist': dict(sorted(Counter(r['n_correct'] for r in rs).items())),
            'truncated_samples': sum(r['n_truncated'] for r in rs),
        })
    return table


def to_markdown(table, k, meta):
    lines = [f'# Re-solving baseline wrong answers ({k} samples each)', '',
             f"baseline: `{meta['pred']}` · model: `{meta['model_path']}` · max_new_tokens {meta['max_new_tokens']}", '',
             '| subject | wrong at baseline | never solved (0/k) | recovered (>=1/k) | mean pass rate | '
             'correct-count histogram | truncated samples |',
             '|---|---|---|---|---|---|---|']
    for t in table:
        hist = ' '.join(f'{c}:{n}' for c, n in t['n_correct_hist'].items())
        lines.append(f"| {t['subject']} | {t['wrong_at_baseline']} | {t['never_solved']} ({t['never_solved_pct']:.0f}%) "
                     f"| {t['recovered']} ({t['recovered_pct']:.0f}%) | {t['mean_pass_rate_pct']:.1f}% | {hist} "
                     f"| {t['truncated_samples']} |")
    lines += ['', '- never solved high -> knowledge gap: augment with external data.',
              '- recovered high -> unstable reasoning: self-generated data / DPO likely more effective.',
              '- histogram `c:n` = n questions answered correctly in c of the k samples.']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--subjects', nargs='+', required=True)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--pred', default='outputs/qwen3vl4b_mmmu_val/predictions.jsonl', help='baseline infer.py output')
    p.add_argument('--output_dir', default='outputs/resample_wrong')
    own, rest = p.parse_known_args()
    args = infer.parse_args(rest + ['--output', os.path.join(own.output_dir, 'resample.jsonl'),
                                    '--subjects', *own.subjects])
    os.makedirs(own.output_dir, exist_ok=True)
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    from datasets import load_dataset
    from transformers import AutoProcessor

    with open(own.pred) as f:
        baseline = [json.loads(line) for line in f]
    wrong = {r['id'] for r in baseline if r['subject'] in own.subjects and not is_correct(r, r['response'])}
    print(f'{len(wrong)} baseline-wrong questions in {own.subjects}')

    processor = AutoProcessor.from_pretrained(args.model_path, revision=args.model_revision or None)
    records, inputs = [], []
    for subject in own.subjects:
        ds = load_dataset('MMMU/MMMU', subject, split='validation', revision=args.dataset_revision,
                          cache_dir=args.data_root)
        for sample in ds:
            if sample['id'] in wrong:
                inp, rec = infer.prepare_sample(sample, subject, processor, args)
                inputs.append(inp)
                records.append(rec)
    assert len(records) == len(wrong), f'{len(wrong) - len(records)} wrong ids not found in the dataset'

    llm = infer.make_llm(args)
    t0 = time.time()
    outputs = llm.generate(inputs, sampling_params=infer.make_sampling(args, n=own.k))
    gen_sec = time.time() - t0

    rows = []
    with open(os.path.join(own.output_dir, 'resample.jsonl'), 'w') as f:
        for rec, out in zip(records, outputs):
            samples = [{'response': o.text, 'finish_reason': o.finish_reason, 'num_output_tokens': len(o.token_ids),
                        'correct': is_correct(rec, o.text)} for o in out.outputs]
            row = dict(rec, samples=samples, n_correct=sum(s['correct'] for s in samples),
                       n_truncated=sum(s['finish_reason'] == 'length' for s in samples))
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            rows.append(row)

    table = summarize(rows, own.k)
    meta = {'pred': own.pred, 'k': own.k, 'model_path': args.model_path, 'max_new_tokens': args.max_new_tokens,
            'generate_seconds': round(gen_sec, 1), 'infer_args': vars(args)}
    with open(os.path.join(own.output_dir, 'summary.json'), 'w') as f:
        json.dump({'meta': meta, 'table': table}, f, indent=2)
    md = to_markdown(table, own.k, meta)
    with open(os.path.join(own.output_dir, 'summary.md'), 'w') as f:
        f.write(md)
    print(md)


if __name__ == '__main__':
    main()
