"""Score infer.py output with one parser. Runs on CPU (numpy only).

  --parser qwen : Qwen3-VL official rule matcher (VLMEvalKit can_infer) + MMMU_preproc for open questions,
                  GPT-judge stage removed -> rule failure counts as wrong.
  --parser mmmu : MMMU official parse_multi_choice_response / parse_open_response / eval_open, unchanged
                  (including its seeded random fallback for unparseable multiple-choice answers).

  --extract final (default): before the parser, cut the response down to its final answer with the fixed rules in
                  extract_final_answer(). Both parsers were written for short answers; on long CoT responses they
                  see every option letter mentioned during reasoning (Qwen's pipeline hands those to a GPT judge).
                  Writes scores_{parser}.*
  --extract none : feed the full response to the parser, as upstream does. Writes scores_{parser}_raw.*
"""
import argparse
import ast
import csv
import json
import os
import random
import re
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from third_party import mmmu_eval_utils as mmmu  # noqa: E402
from third_party.qwen_matching import can_infer  # noqa: E402


# Only markers with a published precedent, nothing tuned to this model's outputs:
#   \boxed{...}         hendrycks/math modeling/dataset/util.py last_boxed_only_string (last one, braces balanced)
#   "answer is" / "Answer:"  TIGER-AI-Lab/MMLU-Pro evaluate_from_api.py extract_answer / extract_again
#                       (r"answer is \(?([A-J])\)?", r'.*[aA]nswer:\s*([A-J])')
_PHRASE = re.compile(r'answer is:?|[aA]nswer:')  # ':?' so "answer is:" behaves like "answer is"
_LETTER = re.compile(r'^\(?([A-Z])\)?(?!\w)')  # MMLU-Pro's \(?([A-J])\)?, plus a word boundary


def _last_boxed(text):
    """(start, content) of the last \\boxed{...}, braces balanced; None if absent or unclosed."""
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    depth, i = 0, start + len('\\boxed')
    for j in range(i, len(text)):
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        if depth == 0:
            return start, text[i + 1:j]
    return None


def _clean(span):
    """Drop markdown emphasis and LaTeX math/text wrappers: '**$\\text{C}$**' -> 'C'."""
    s = re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}', r'\1', span)
    return re.sub(r'\*\*|__|\$', '', s)


def extract_final_answer(response, options):
    """Fixed, model-independent rule that stands in for Qwen's GPT-judge extraction step. Returns (text, how).

    1. span = content of the last \\boxed{} or the rest of the line after the last "answer is" / "Answer:"
       (markdown emphasis dropped, then leading whitespace incl. newlines skipped as in MMLU-Pro's \\s*),
       whichever marker comes later.
    2. multiple-choice only: a span starting with an option label ('C', 'C.', '(C)') becomes that letter.
    3. No marker (or an empty span) -> the full response goes to the parser unchanged, i.e. upstream behavior.
    """
    text = response.split('</think>')[-1].strip()
    cands = []
    boxed = _last_boxed(text)
    if boxed:
        cands.append((boxed[0], boxed[1], 'boxed'))
    phrases = list(_PHRASE.finditer(text))
    if phrases:
        m = phrases[-1]
        cands.append((m.start(), _clean(text[m.end():]).lstrip().split('\n')[0], 'phrase'))
    if not cands:
        return response, 'none'
    _, span, how = max(cands, key=lambda c: c[0])
    span = _clean(span).strip()
    if not span:
        return response, 'none'
    if options:
        m = _LETTER.match(span)
        if m and m.group(1) in string.ascii_uppercase[:len(options)]:
            return m.group(1), how
    return span, how


def score_qwen(rec, text=None):
    """Mirrors run_mmmu.py run_evaluation() + eval_utils.extract_answer_from_item() without the judge."""
    prediction = (rec['response'] if text is None else text).split('</think>')[-1].strip()
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


def score_mmmu(rec, text=None):
    """Mirrors MMMU main_parse_and_eval.py: parse then eval_multi_choice / eval_open."""
    response = rec['response'] if text is None else text
    if rec['options']:
        index2ans = dict(zip(string.ascii_uppercase, rec['options']))
        before = _CountingRandom.calls
        pred = mmmu.parse_multi_choice_response(response, list(index2ans), index2ans)
        method = 'random' if _CountingRandom.calls > before else 'rule'
        return pred, method, mmmu.eval_multi_choice(rec['answer'], pred)
    gold = rec['answer']
    if gold.startswith('['):  # HF stores multi-answer gold as a list literal, e.g. "['Tampa', 'Florida']"
        gold = ast.literal_eval(gold)
    pred = mmmu.parse_open_response(response)
    return json.dumps(pred, ensure_ascii=False, default=str), 'open', mmmu.eval_open(gold, pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True, help='JSONL from infer.py')
    p.add_argument('--parser', choices=['qwen', 'mmmu'], required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--extract', choices=['final', 'none'], default='final',
                   help='final: parse extract_final_answer() output (scores_{parser}.*); '
                        'none: parse the full response like upstream (scores_{parser}_raw.*)')
    args = p.parse_args()

    with open(args.pred) as f:
        recs = [json.loads(line) for line in f]
    scorer = score_qwen if args.parser == 'qwen' else score_mmmu

    rows = []
    for rec in recs:
        text, how = extract_final_answer(rec['response'], rec['options']) if args.extract == 'final' else (None, '')
        pred, method, correct = scorer(rec, text)
        # full responses stay in predictions.jsonl (join on id)
        rows.append({'id': rec['id'], 'subject': rec['subject'], 'question_type': rec['question_type'],
                     'answer': rec['answer'], 'extract': how, 'extracted': text, 'parsed': pred, 'method': method,
                     'correct': bool(correct), 'finish_reason': rec['finish_reason']})

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
    extracts = {e: sum(r['extract'] == e for r in rows) for e in dict.fromkeys(r['extract'] for r in rows)}

    os.makedirs(args.output_dir, exist_ok=True)
    name = args.parser if args.extract == 'final' else f'{args.parser}_raw'
    stem = os.path.join(args.output_dir, f'scores_{name}')
    with open(stem + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(stem + '.json', 'w') as f:
        json.dump({'parser': args.parser, 'extract': args.extract, 'macro_acc': macro, 'micro_acc': micro,
                   'num_samples': len(rows), 'methods': methods, 'extracts': extracts, 'subjects': table},
                  f, indent=2)
    lines = ['| No. | Subject | Data Num | Acc |', '|---|---|---|---|']
    lines += [f"| {i} | {t['subject']} | {t['n']} | {t['acc']:.2f} |" for i, t in enumerate(table, 1)]
    lines.append(f"| | **Overall (macro avg)** | **{len(rows)}** | **{macro:.2f}** |")
    with open(stem + '.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print(f'parser={args.parser} extract={args.extract} macro={macro:.2f} micro={micro:.2f} '
          f'methods={methods} extracts={extracts}')


if __name__ == '__main__':
    main()
