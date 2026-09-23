"""Score infer.py output with one parser. Runs on CPU (numpy only).

  --parser qwen : Qwen3-VL official rule matcher (VLMEvalKit can_infer) + MMMU_preproc for open questions,
                  GPT-judge stage removed -> rule failure counts as wrong.
  --parser mmmu : MMMU official parse_multi_choice_response / parse_open_response / eval_open, unchanged
                  (including its seeded random fallback for unparseable multiple-choice answers).
"""
import argparse
import ast
import csv
import json
import os
import random
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from third_party import mmmu_eval_utils as mmmu  # noqa: E402
from third_party.qwen_matching import can_infer  # noqa: E402


def score_qwen(rec):
    """Mirrors run_mmmu.py run_evaluation() + eval_utils.extract_answer_from_item() without the judge."""
    prediction = rec['response'].split('</think>')[-1].strip()
    # MMMU_preproc: open question -> 2-choice {A: gold, B: 'Other Answers'}
    choices = (dict(zip(string.ascii_uppercase, rec['options'])) if rec['options']
               else {'A': rec['answer'], 'B': 'Other Answers'})
    # answer_map: any gold that is not a single uppercase letter becomes 'A' (applied to all questions upstream)
    gt = rec['answer'] if rec['answer'] in list(string.ascii_uppercase) else 'A'
    ret = can_infer(prediction, dict(choices))
    if ret == 'Z':
        return 'Z', 'rule_Z', False  # Qwen returns 'Z' as the answer (never a hit)
    if not ret:
        return '', 'fail', False  # Qwen would call the GPT judge here
    return ret, 'rule', ret == gt


class _CountingRandom:
    """Stand-in for the vendored module's `random`, to count MMMU's random fallback without editing it."""
    calls = 0

    def choice(self, seq):
        _CountingRandom.calls += 1
        return random.choice(seq)  # stdlib random, already seeded(42) by the vendored module


mmmu.random = _CountingRandom()


def score_mmmu(rec):
    """Mirrors MMMU main_parse_and_eval.py: parse then eval_multi_choice / eval_open."""
    if rec['options']:
        index2ans = dict(zip(string.ascii_uppercase, rec['options']))
        before = _CountingRandom.calls
        pred = mmmu.parse_multi_choice_response(rec['response'], list(index2ans), index2ans)
        method = 'random' if _CountingRandom.calls > before else 'rule'
        return pred, method, mmmu.eval_multi_choice(rec['answer'], pred)
    gold = rec['answer']
    if gold.startswith('['):  # HF stores multi-answer gold as a list literal, e.g. "['Tampa', 'Florida']"
        gold = ast.literal_eval(gold)
    pred = mmmu.parse_open_response(rec['response'])
    return json.dumps(pred, ensure_ascii=False, default=str), 'open', mmmu.eval_open(gold, pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True, help='JSONL from infer.py')
    p.add_argument('--parser', choices=['qwen', 'mmmu'], required=True)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    with open(args.pred) as f:
        recs = [json.loads(line) for line in f]
    scorer = score_qwen if args.parser == 'qwen' else score_mmmu

    rows = []
    for rec in recs:
        pred, method, correct = scorer(rec)
        rows.append({'id': rec['id'], 'subject': rec['subject'], 'question_type': rec['question_type'],
                     'answer': rec['answer'], 'parsed': pred, 'method': method, 'correct': bool(correct),
                     'finish_reason': rec['finish_reason'], 'response': rec['response']})

    subjects = list(dict.fromkeys(r['subject'] for r in rows))
    table = []
    for s in subjects:
        rs = [r for r in rows if r['subject'] == s]
        table.append({'subject': s, 'n': len(rs), 'correct': sum(r['correct'] for r in rs),
                      'acc': 100 * sum(r['correct'] for r in rs) / len(rs),
                      'not_rule': sum(r['method'] in ('fail', 'rule_Z', 'random') for r in rs),
                      'truncated': sum(r['finish_reason'] == 'length' for r in rs)})
    macro = sum(t['acc'] for t in table) / len(table)
    micro = 100 * sum(r['correct'] for r in rows) / len(rows)
    methods = {m: sum(r['method'] == m for r in rows) for m in dict.fromkeys(r['method'] for r in rows)}

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.join(args.output_dir, f'scores_{args.parser}')
    with open(stem + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(stem + '.json', 'w') as f:
        json.dump({'parser': args.parser, 'macro_acc': macro, 'micro_acc': micro, 'num_samples': len(rows),
                   'methods': methods, 'subjects': table}, f, indent=2)
    lines = ['| No. | Subject | Data Num | Acc |', '|---|---|---|---|']
    lines += [f"| {i} | {t['subject']} | {t['n']} | {t['acc']:.2f} |" for i, t in enumerate(table, 1)]
    lines.append(f"| | **Overall (macro avg)** | **{len(rows)}** | **{macro:.2f}** |")
    with open(stem + '.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print(f'parser={args.parser} macro={macro:.2f} micro={micro:.2f} methods={methods}')


if __name__ == '__main__':
    main()
