"""Loop signals in truncated responses and branch points for dpo_resample.py. CPU only.

What the gold set (loop_gold.html, 17 labels on 2026-11-17) showed, and how this file uses it:
  - every labeled truncated response was a loop (17/17, types A1 A2 B C D E; no G "long calculation" yet), while
    the old rule (classify_errors) caught 12 and the repeat rule below 14: the misses are A2 loops that re-state a
    hypothesis in new words. So dpo_resample.py takes ALL truncated responses; the rules are recorded, not used to gate.
  - no text rule found the human onset well (repeat rule: 0/14 within 300 chars, errors up to +-17k chars; the
    density rule below: about half within 2k chars). A loop onset is a judgment call ("Wait - perhaps ..." starts
    long before verbatim repetition). So branch points do not depend on an onset: they are fixed fractions of the
    response, and dpo_resample.py keeps the latest one from which the model actually escapes.

Signals per response (positions are character offsets, the unit of loop_gold.html labels):
  units      lines; long lines cut at sentence ends, then into <= 200-char chunks
  repeat     a unit whose normalized text appeared before, or >= --shingle_frac of its word 5-grams did
  restart    a unit starting with a hesitation/restart phrase ("Wait", "Perhaps", "Alternatively", "But maybe", ...)
  onset_char     (repeat rule) first repeat unit after which >= --tail_frac of the remaining chars are repeats
  dense_char     (density rule) first repeat/restart unit where a 2,000-char window is >= 70% repeat/restart chars
                 and stays >= 60% on average until the end
  branches       line starts at --fractions of the response (default 60/45/30/15 %), latest first

  python scripts/loop_onset.py                                    # train_final truncations -> outputs/dpo/loop_candidates.jsonl
  python scripts/loop_onset.py --compare_gold outputs/loop_gold/loop_gold_labels.csv
"""
import argparse
import csv
import json
import os
import re
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_SENT = re.compile(r'(?<=[.!?。])\s+')
_WORD = re.compile(r'\S+')


def units(text, max_line=400, chunk=200):
    """[(start, end, text)] covering the response: lines; lines longer than max_line split at sentence ends,
    sentences still longer than max_line cut into chunk-sized pieces."""
    out, pos = [], 0
    for line in text.split('\n'):
        if len(line) <= max_line:
            out.append((pos, pos + len(line), line))
        else:
            p = 0
            for sent in _SENT.split(line):
                i = line.find(sent, p)
                p = i + len(sent)
                step = chunk if len(sent) > max_line else max(len(sent), 1)
                for k in range(0, max(len(sent), 1), step):
                    piece = sent[k:k + step]
                    out.append((pos + i + k, pos + i + k + len(piece), piece))
        pos += len(line) + 1
    return out


def _norm(s):
    return ' '.join(s.lower().split())


def _shingles(words, n=5):
    return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]


RESTART = re.compile(
    r"^[\W_]*(wait|perhaps|alternatively|hmm|maybe|actually|unless|or perhaps|or maybe|another (idea|possibility|"
    r"approach|thought|interpretation)|let me (re|try|check|think|consider|look|assume|go back)|"
    r"let[’']s (re|try|assume|look|go back|consider)|but (wait|perhaps|maybe|what if)|what if|hold on|no[,\s—-])",
    re.I)


def unit_flags(text, shingle_frac=0.6, min_unit=15):
    """[(start, end, repeat, restart, src_unit)] for every unit; repeat looks only at text before the unit."""
    us = units(text)
    seen_text, seen_sh, first_where = set(), set(), {}
    out = []
    for i, (a, b, s) in enumerate(us):
        t = _norm(s)
        sh = _shingles(_WORD.findall(t))
        r, origin = False, None
        if len(t) >= min_unit:
            if t in seen_text:
                r, origin = True, first_where.get(t)
            elif sh and sum(x in seen_sh for x in sh) / len(sh) >= shingle_frac:
                r = True
                origin = min((first_where[x] for x in sh if x in first_where), default=None)
        out.append((a, b, r, len(t) >= 10 and bool(RESTART.match(s.strip())), origin))
        if len(t) >= min_unit:
            seen_text.add(t)
            first_where.setdefault(t, i)
        for x in sh:
            seen_sh.add(x)
            first_where.setdefault(x, i)
    return out


def find_onset(text, shingle_frac=0.6, tail_frac=0.7, min_tail=1000, min_unit=15, flags=None):
    """Repeat rule: first repeat unit after which >= tail_frac of the remaining characters are repeats."""
    fl = flags or unit_flags(text, shingle_frac, min_unit)
    n = len(fl)
    rep_chars, all_chars = [0] * (n + 1), [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        w = fl[i][1] - fl[i][0]
        all_chars[i] = all_chars[i + 1] + w
        rep_chars[i] = rep_chars[i + 1] + (w if fl[i][2] else 0)
    for i in range(n):
        if fl[i][2] and all_chars[i] >= min_tail and rep_chars[i] / all_chars[i] >= tail_frac:
            src = fl[i][4]
            return {'loop': True, 'onset_char': fl[i][0], 'first_copy_char': fl[src][0] if src is not None else None,
                    'tail_repeat_frac': round(rep_chars[i] / all_chars[i], 3)}
    return {'loop': False, 'onset_char': None, 'first_copy_char': None,
            'tail_repeat_frac': round(rep_chars[0] / max(all_chars[0], 1), 3)}


def dense_onset(text, window=2000, start_frac=0.7, stay_frac=0.6, min_tail=1000, flags=None):
    """Density rule: first repeat/restart unit whose window is >= start_frac repeat/restart chars and whose
    following windows stay >= stay_frac on average. None if the response never gets there."""
    fl = flags or unit_flags(text)
    n = len(fl)
    w = [(b - a, (b - a) if (r or m) else 0) for a, b, r, m, _ in fl]
    win, j, tot, good = [], 0, 0, 0
    for i in range(n):
        while j < n and tot < window:
            tot += w[j][0]
            good += w[j][1]
            j += 1
        win.append(good / max(tot, 1))
        tot -= w[i][0]
        good -= w[i][1]
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + win[i]
    for i, (a, b, r, m, _) in enumerate(fl):
        if (r or m) and win[i] >= start_frac and suffix[i] / (n - i) >= stay_frac and len(text) - a >= min_tail:
            return a
    return None


def loop_share(flags, start, end):
    """Share of characters in [start, end) that belong to repeat/restart units (judged with the whole text before)."""
    tot = good = 0
    for a, b, r, m, _ in flags:
        x = max(0, min(b, end) - max(a, start))
        tot += x
        good += x if (r or m) else 0
    return good / tot if tot else 0.0


def line_start_before(text, pos):
    """Start of the line containing pos (a prefix that ends right after a newline)."""
    return text.rfind('\n', 0, max(pos, 0)) + 1


def branch_candidates(text, fractions=(0.6, 0.45, 0.3, 0.15), min_prefix=200):
    """Line starts at fixed fractions of the response, latest first."""
    pts = {line_start_before(text, int(len(text) * f)) for f in fractions}
    return sorted((p for p in pts if p >= min_prefix), reverse=True)


# ----------------------------------------------------------------------------- CLI
def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def scan(a):
    from classify_errors import repeated_line_ratio, zlib_ratio
    keep = set().union(*({r['id'] for r in read_jsonl(f)} for f in a.ids)) if a.ids else None
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    n = n_loop = n_old = both = n_dense = 0
    with open(a.out, 'w', encoding='utf-8') as f:
        for path in a.resp:
            for g in read_jsonl(path):
                if keep is not None and g['id'] not in keep:
                    continue
                for k, s in enumerate(g['samples']):
                    if s['finish_reason'] != 'length':
                        continue
                    text = s['response']
                    fl = unit_flags(text, a.shingle_frac)
                    r = find_onset(text, a.shingle_frac, a.tail_frac, a.min_tail, flags=fl)
                    r['dense_char'] = dense_onset(text, flags=fl)
                    old = zlib_ratio(text) < 0.10 or repeated_line_ratio(text) >= 0.8
                    n += 1
                    n_loop += r['loop']
                    n_old += old
                    both += r['loop'] and old
                    n_dense += r['dense_char'] is not None
                    r['branches'] = branch_candidates(text, a.fractions)
                    f.write(json.dumps({'id': g['id'], 'sample': k, 'source': os.path.basename(path), 'chars': len(text),
                                        'old_rule_loop': old, **r}, ensure_ascii=False) + '\n')
    print(f'truncated {n} (all kept, with branch points): repeat rule loop {n_loop} ({n_loop / max(n, 1):.0%}), '
          f'old rule loop {n_old}, both {both}, density rule found {n_dense}')
    print(f'wrote {a.out}')


def compare_gold(a):
    """Labels from loop_gold.html vs this rule (and the old rule from loop_gold_key.csv)."""
    with open(a.compare_gold, encoding='utf-8-sig') as f:
        labels = {r['key']: r for r in csv.DictReader(f)}
    with open(a.gold_key, encoding='utf-8') as f:
        key = {r['key']: r for r in csv.DictReader(f)}
    resp = {}
    for path in a.resp:
        for g in read_jsonl(path):
            for k, s in enumerate(g['samples']):
                resp[(g['id'], k)] = s['response']
    rows, errs = [], []
    for k, lab in sorted(labels.items()):
        if lab.get('is_loop') not in ('loop', 'no'):
            continue
        kk = key[k]
        text = resp[(kk['id'], int(kk['sample']))]
        assert len(text) == int(kk['chars']), f'{k}: response differs from the one labeled'
        fl = unit_flags(text, a.shingle_frac)
        r = find_onset(text, a.shingle_frac, a.tail_frac, a.min_tail, flags=fl)
        dn = dense_onset(text, flags=fl)
        human = lab['is_loop'] == 'loop'
        rows.append((k, human, kk['pool'] == 'flagged', r['loop'], lab.get('loop_type') or '-'))
        if human and lab.get('onset_char'):
            h = int(lab['onset_char'])
            errs.append((k, r['onset_char'] - h if r['loop'] else None, dn - h if dn is not None else None,
                         [round(b / len(text), 2) for b in branch_candidates(text, a.fractions) if b >= h]))
    def table(col):
        tp = sum(r[1] and r[col] for r in rows)
        fp = sum((not r[1]) and r[col] for r in rows)
        fn = sum(r[1] and not r[col] for r in rows)
        return tp, fp, fn
    print(f'labeled (loop / not loop): {len(rows)}  (unsure and blank skipped)')
    for name, col in (('old rule (classify_errors)', 2), ('new rule (loop_onset)', 3)):
        tp, fp, fn = table(col)
        print(f'  {name}: caught {tp}, false alarm {fp}, missed {fn}')
    for name, col in (('repeat rule onset', 1), ('density rule onset', 2)):
        d = [e[col] for e in errs if e[col] is not None]
        if d:
            print(f'{name} - human (chars): found {len(d)}/{len(errs)}, median {median(d):+.0f}, '
                  f'within 300 {sum(abs(x) <= 300 for x in d)}, within 2000 {sum(abs(x) <= 2000 for x in d)}')
    inside = [len(e[3]) for e in errs]
    print(f'branch points at or after the human onset (inside the loop) per response: {inside}')
    print('by labeled type (n: old rule caught / new rule caught):')
    for t in sorted({r[4] for r in rows}):
        rs = [r for r in rows if r[4] == t]
        print(f'  {t}: n={len(rs)} human-loop={sum(r[1] for r in rs)} old={sum(r[2] for r in rs)} new={sum(r[3] for r in rs)}')
    missed = [r[0] for r in rows if r[1] and not r[3]]
    if missed:
        print(f'  new rule missed: {missed}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--resp', nargs='+', default=['outputs/rl_rollout/rollouts_resp.jsonl',
                                                 'outputs/rl_rollout/pilot_resp.jsonl'])
    p.add_argument('--ids', nargs='*', default=['data/rl/train_final.jsonl'])
    p.add_argument('--out', default='outputs/dpo/loop_candidates.jsonl')
    p.add_argument('--shingle_frac', type=float, default=0.6)
    p.add_argument('--tail_frac', type=float, default=0.7)
    p.add_argument('--min_tail', type=int, default=1000)
    p.add_argument('--fractions', type=float, nargs='+', default=[0.6, 0.45, 0.3, 0.15])
    p.add_argument('--compare_gold', default=None, help='loop_gold_labels.csv exported from loop_gold.html')
    p.add_argument('--gold_key', default='outputs/loop_gold/loop_gold_key.csv')
    a = p.parse_args()
    compare_gold(a) if a.compare_gold else scan(a)


if __name__ == '__main__':
    main()
