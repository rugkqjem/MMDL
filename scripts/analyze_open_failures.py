"""Group the Qwen parser's open-ended failures by *why* the rule failed. Runs on CPU (stdlib only).

Diagnostic tool, not part of scoring: it re-reads a finished run (scores_qwen.csv + predictions.jsonl)
and splits every wrong open-ended row into surface-notation mismatches (LaTeX markup, thousands
separators, rounding) versus genuinely wrong values, so the gap analysis can say how much of the
'rule failure = wrong' policy costs. Nothing here feeds back into the score, and
scripts/third_party/ is imported unchanged.

Upstream, a rule failure goes to a GPT-3.5 judge (see score.py); we count it wrong. The categories
below are the judge's decisions approximated with explicit, inspectable rules.

  latex_only      value equals gold, the span only differs by LaTeX markup (\\dfrac, \\sqrt, ...)
  format_only     value equals gold, differs by non-LaTeX surface form (commas, case, trailing words)
  rounding_near   value within 5% of gold but not equal at gold's precision
  wrong_value     the model's value is a different value (LaTeX markup may still be present)
  no_marker       extract_final_answer() found no marker, so the full response went to the parser
  no_final_answer response hit max_new_tokens and never stated an answer

For each row it also probes scripts/third_party's can_infer with progressively normalized spans, to
record which single normalization step would flip the row to correct:
  delatex -> round to gold's precision -> unwrap HF's list-literal gold ("['24/7', '3.429']").
"""
import argparse
import ast
import csv
import json
import math
import os
import re
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from third_party.qwen_matching import can_infer  # noqa: E402

csv.field_size_limit(10 ** 9)

# LaTeX commands that show up in this model's final-answer spans
LATEX_CMD = re.compile(r'\\(?:dfrac|tfrac|frac|sqrt|times|cdot|approx|text|mathrm|left|right|Omega|mu|pi|,|;|!|\s)')
# units and magnitude words that the gold answers never carry
_UNITS = re.compile(r'\b(?:k?(?:ohm|Omega)s?|[munk]?[AVWFHJ]|Hz|kHz|MHz|mm|cm|m|km|g|kg|mol|L|mL|s|ms'
                    r'|years?|million|billion|unfavorable|favorable|rad|dB)\b')
CATEGORIES = ['latex_only', 'format_only', 'rounding_near', 'wrong_value', 'no_marker', 'no_final_answer']


def _strip_wrappers(s):
    """Drop LaTeX spacing/formatting wrappers and units, keeping the arithmetic intact."""
    s = re.sub(r'\\(?:left|right|,|;|!|\ )', ' ', s)
    s = re.sub(r'\\(?:text|mathrm|mathbf)\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\(?:Omega|ohm|mu|circ|degree|percent)', ' ', s)
    return _UNITS.sub(' ', s)


def delatex(span):
    """Resolve LaTeX notation to plain arithmetic. Changes notation only, never the value."""
    s = _strip_wrappers(span)
    s = re.sub(r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', s)
    s = re.sub(r'(?<=[\d)])\s*(?=\\sqrt)', '*', s)  # 2\sqrt{2} -> 2*\sqrt{2}
    s = re.sub(r'\\sqrt\{?(\d+(?:\.\d+)?)\}?', lambda m: f'{float(m.group(1)) ** 0.5:.10g}', s)
    s = s.replace(r'\times', '*').replace(r'\cdot', '*').replace(r'\approx', ' ').replace('$', '')
    return re.sub(r'\s+', ' ', s).strip()


_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)


def _arith(expr):
    """Evaluate a pure-arithmetic string without eval(). Returns a float, or None if not arithmetic."""
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        return float(_fold(tree.body))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


def _fold(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        v = _fold(node.operand)
        return -v if isinstance(node.op, ast.USub) else +v
    a, b = _fold(node.left), _fold(node.right)
    return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
            ast.Div: lambda: a / b, ast.Pow: lambda: a ** b}[type(node.op)]()


def to_float(span):
    """Best-effort numeric value of an answer span (LaTeX, units, thousands separators allowed)."""
    s = delatex(span).strip().rstrip('.').replace('%', '')
    s = re.sub(r'10\s*\^\s*\{?(-?\d+)\}?', r'(10**\1)', s)
    s = re.sub(r'\^\s*\{?(-?\d+)\}?', r'**\1', s)
    s = re.sub(r'(?<=\d),(?=\d{3}\b)', '', s)       # thousands separator
    s = re.sub(r'(?<=[\d)])\s*(?=\()', '*', s)      # 2(10**3) -> 2*(10**3)
    return _arith(s) if s else None


def golds(raw):
    """HF stores multi-answer gold as a list literal, e.g. "['24/7', '3.429']"."""
    if raw.startswith('['):
        try:
            return [str(x) for x in ast.literal_eval(raw)]
        except (ValueError, SyntaxError):
            pass
    return [raw]


def _squash(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def value_matches(span, gold_list):
    """True if the span states one of the gold values, ignoring notation and gold's precision."""
    v = to_float(span)
    for g in gold_list:
        if _squash(g) and _squash(span) == _squash(g):
            return True
        gv = to_float(g)
        if v is None or gv is None:
            continue
        if math.isclose(v, gv, rel_tol=1e-9, abs_tol=1e-12):
            return True
        dp = len(g.split('.')[1]) if '.' in g else 0   # gold carries the intended precision
        if math.isclose(round(v, dp), gv, rel_tol=1e-9, abs_tol=1e-12):
            return True
    return False


def accepted(span, rec, gold_override=None):
    """score_qwen()'s verdict for an arbitrary span: does the unmodified rule call it correct?"""
    gold = rec['answer'] if gold_override is None else gold_override
    choices = (dict(zip(string.ascii_uppercase, rec['options'])) if rec['options']
               else {'A': gold, 'B': 'Other Answers'})
    gt = rec['answer'] if rec['answer'] in list(string.ascii_uppercase) else 'A'
    ret = can_infer(span, dict(choices))
    return bool(ret) and ret != 'Z' and ret == gt


def recoverable_by(span, rec, gold_list):
    """First normalization step (if any) that makes the unmodified rule accept the span."""
    if accepted(span, rec):
        return 'already_correct'
    d = delatex(span)
    if accepted(d, rec):
        return 'delatex'
    v = to_float(d)
    if v is not None:
        for g in gold_list:
            dp = len(g.split('.')[1]) if '.' in g else 0
            if accepted(f'{round(v, dp):.{dp}f}', rec):
                return 'delatex+round'
    if rec['answer'].startswith('['):  # list-literal gold never matches as a raw string
        for g in gold_list:
            for cand in (d, f'{round(v, len(g.split(".")[1]) if "." in g else 0)}' if v is not None else d):
                if accepted(cand, rec, gold_override=g):
                    return 'delatex+unwrap_list_gold'
    return ''


def classify(row, gold_list):
    span = row['extracted'].strip()
    if row['finish_reason'] == 'length':
        return 'no_final_answer'
    if row['extract'] == 'none':
        return 'no_marker'
    if value_matches(span, gold_list):
        return 'latex_only' if LATEX_CMD.search(span) else 'format_only'
    v, gv = to_float(span), to_float(gold_list[0])
    if v is not None and gv not in (None, 0) and math.isclose(v, gv, rel_tol=0.05):
        return 'rounding_near'
    return 'wrong_value'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', required=True, help='directory holding predictions.jsonl and scores_qwen.csv')
    p.add_argument('--output_dir', help='where to write open_failures.{csv,md} (default: --run_dir)')
    args = p.parse_args()
    out_dir = args.output_dir or args.run_dir

    with open(os.path.join(args.run_dir, 'scores_qwen.csv')) as f:
        scored = list(csv.DictReader(f))
    with open(os.path.join(args.run_dir, 'predictions.jsonl')) as f:
        recs = {r['id']: r for r in map(json.loads, f)}

    rows = []
    for row in scored:
        if row['question_type'] == 'multiple-choice' or row['correct'] == 'True':
            continue
        rec, gold_list = recs[row['id']], golds(row['answer'])
        span = row['extracted'].strip()
        rows.append({'id': row['id'], 'subject': row['subject'], 'category': classify(row, gold_list),
                     'gold': row['answer'], 'extracted': span, 'extract': row['extract'],
                     'finish_reason': row['finish_reason'], 'method': row['method'],
                     'has_latex': bool(LATEX_CMD.search(span)), 'value': to_float(span),
                     'gold_value': to_float(gold_list[0]), 'recoverable_by': recoverable_by(span, rec, gold_list),
                     'response_chars': len(rec['response'])})
    rows.sort(key=lambda r: (CATEGORIES.index(r['category']), r['id']))

    n_open = sum(r['question_type'] != 'multiple-choice' for r in scored)
    counts = {c: sum(r['category'] == c for r in rows) for c in CATEGORIES}
    n_subjects = len({r['subject'] for r in scored})
    per_subject = len(scored) // n_subjects
    notation = counts['latex_only'] + counts['format_only']
    # one extra correct answer moves its subject by 100/per_subject points, and the macro average by 1/n_subjects of that
    macro_gain = notation * (100 / per_subject) / n_subjects

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'open_failures.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [f'# Open-ended failures, Qwen parser ({len(rows)} of {n_open} open questions wrong)', '',
             '| Category | N |', '|---|---|']
    lines += [f'| {c} | {counts[c]} |' for c in CATEGORIES]
    lines += ['', f'Notation-only failures (latex_only + format_only): **{notation}** '
                  f'-> at most **+{macro_gain:.2f}** macro points if all were judged correct.', '']
    for c in CATEGORIES:
        group = [r for r in rows if r['category'] == c]
        if not group:
            continue
        lines += [f'## {c} ({len(group)})', '', '| id | gold | extracted | recoverable by |', '|---|---|---|---|']
        for r in group:
            span = r['extracted'].replace('|', '\\|').replace('\n', ' ')
            span = span if len(span) <= 80 else span[:77] + '...'
            lines.append(f"| {r['id']} | `{r['gold']}` | `{span}` | {r['recoverable_by'] or '-'} |")
        lines.append('')
    with open(os.path.join(out_dir, 'open_failures.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print(f'wrote open_failures.csv / open_failures.md to {out_dir}')


if __name__ == '__main__':
    main()
