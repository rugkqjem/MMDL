"""Collect scores_qwen.json from several eval output dirs into one comparison table (CPU, no inference).

    python scripts/compare_ablation.py --runs baseline=outputs/ablation/eval_baseline deepstack=outputs/ablation/eval_deepstack \
        --output outputs/ablation/comparison.md
The first run is the reference for the delta column.
"""
import argparse
import json
import os


def load(run_dir, parser):
    with open(os.path.join(run_dir, f'scores_{parser}.json')) as f:
        return json.load(f)


def table(runs, parser):
    scores = {name: load(d, parser) for name, d in runs}
    ref = scores[runs[0][0]]['macro_acc']
    lines = [f'## Overall (parser={parser}, extract=final, macro average of 30 subjects)', '',
             '| run | overall | delta | correct | truncated |', '|---|---|---|---|---|']
    for name, s in scores.items():
        correct = sum(x['correct'] for x in s['subjects'])
        truncated = sum(x['truncated'] for x in s['subjects'])
        lines.append(f"| {name} | {s['macro_acc']:.2f} | {s['macro_acc'] - ref:+.2f} | {correct}/{s['num_samples']} "
                     f'| {truncated} |')
    lines += ['', '## Per subject (acc %)', '', '| subject | ' + ' | '.join(scores) + ' |',
              '|---|' + '---|' * len(scores)]
    for i, subj in enumerate(scores[runs[0][0]]['subjects']):
        lines.append(f"| {subj['subject']} | " + ' | '.join(f"{s['subjects'][i]['acc']:.1f}" for s in scores.values())
                     + ' |')
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True, help='name=eval_output_dir, reference first')
    p.add_argument('--parser', default='qwen', choices=['qwen', 'mmmu'])
    p.add_argument('--output', default=None, help='also write the table to this .md file')
    args = p.parse_args()
    md = table([tuple(r.split('=', 1)) for r in args.runs], args.parser)
    print(md)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(md)


if __name__ == '__main__':
    main()
