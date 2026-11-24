"""Compare two models scored by scripts/score.py on MMMU-Pro standard (10 options). CPU only, stdlib only.

    python mmmu_pro_eval/compare.py --a outputs/mmmu_pro_std10/baseline --b outputs/mmmu_pro_std10/v1b \
        --name_a baseline --name_b v1b --out outputs/mmmu_pro_std10/compare

Reads <dir>/scores_qwen.csv (default --scores scores_qwen = Qwen parser + final-answer extraction, the repo's
reported score). Overall = micro accuracy over all questions (MMMU-Pro reports overall this way); macro over
subjects is shown too. Paired comparison on the same question ids: exact two-sided sign test on the
discordant pairs, for correctness and for truncation (finish_reason == 'length').
"""
import argparse
import csv
import json
import os
from math import comb


def load(d, scores):
    with open(os.path.join(d, scores + '.csv'), newline='') as f:
        return {r['id']: r for r in csv.DictReader(f)}


def sign_test(k, n):
    """Exact two-sided binomial test, p = 0.5."""
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(rows):
    subj = {}
    for r in rows.values():
        s = subj.setdefault(r['subject'], [0, 0, 0])
        s[0] += 1
        s[1] += r['correct'] == 'True'
        s[2] += r['finish_reason'] == 'length'
    n = len(rows)
    micro = 100 * sum(v[1] for v in subj.values()) / n
    macro = sum(100 * v[1] / v[0] for v in subj.values()) / len(subj)
    return {'n': n, 'correct': sum(v[1] for v in subj.values()), 'micro': micro, 'macro': macro,
            'truncated': sum(v[2] for v in subj.values()),
            'not_rule': sum(r['method'] in ('fail', 'rule_Z', 'random') for r in rows.values()),
            'subjects': subj}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True, help='score dir of model A (reference, e.g. baseline)')
    p.add_argument('--b', required=True, help='score dir of model B (e.g. v1b)')
    p.add_argument('--name_a', default='baseline')
    p.add_argument('--name_b', default='v1b')
    p.add_argument('--scores', default='scores_qwen', help='score file stem written by score.py')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    A, B = load(args.a, args.scores), load(args.b, args.scores)
    if set(A) != set(B):
        raise SystemExit(f'question ids differ: {len(set(A) - set(B))} only in A, {len(set(B) - set(A))} only in B')
    sa, sb = summarize(A), summarize(B)
    ids = sorted(A)
    gain = sum(A[i]['correct'] == 'False' and B[i]['correct'] == 'True' for i in ids)
    loss = sum(A[i]['correct'] == 'True' and B[i]['correct'] == 'False' for i in ids)
    tfix = sum(A[i]['finish_reason'] == 'length' and B[i]['finish_reason'] != 'length' for i in ids)
    tnew = sum(A[i]['finish_reason'] != 'length' and B[i]['finish_reason'] == 'length' for i in ids)
    paired = {'b_gains': gain, 'b_losses': loss, 'p_correct': sign_test(gain, gain + loss),
              'trunc_fixed': tfix, 'trunc_new': tnew, 'p_trunc': sign_test(tfix, tfix + tnew)}

    na, nb = args.name_a, args.name_b
    L = [f'# MMMU-Pro standard (10 options): {na} vs {nb}', '',
         f'Scores: `{args.scores}` · questions: {sa["n"]}', '',
         f'| Model | Overall (micro) | Macro (subjects) | Correct | Truncated | Unparsed |', '|---|---|---|---|---|---|']
    for name, s in ((na, sa), (nb, sb)):
        L.append(f'| {name} | {s["micro"]:.2f} | {s["macro"]:.2f} | {s["correct"]}/{s["n"]} | {s["truncated"]} | '
                 f'{s["not_rule"]} |')
    L += ['', f'Δ ({nb} − {na}): {sb["micro"] - sa["micro"]:+.2f} pt (micro), {sb["macro"] - sa["macro"]:+.2f} pt (macro)',
          '', '## Same-question comparison (exact sign test)', '',
          '| | +/− | p |', '|---|---|---|',
          f'| Correct ({nb} right & {na} wrong / reverse) | +{gain} / −{loss} | {paired["p_correct"]:.3g} |',
          f'| Truncation (fixed / new) | {tfix} / {tnew} | {paired["p_trunc"]:.3g} |',
          '', '## Per subject', '',
          f'| Subject | N | {na} | {nb} | Δ | {na} trunc | {nb} trunc |', '|---|---|---|---|---|---|---|']
    subjects = sorted(sa['subjects'])
    per = []
    for s in subjects:
        a, b = sa['subjects'][s], sb['subjects'][s]
        acc_a, acc_b = 100 * a[1] / a[0], 100 * b[1] / b[0]
        per.append({'subject': s, 'n': a[0], 'acc_a': acc_a, 'acc_b': acc_b, 'trunc_a': a[2], 'trunc_b': b[2]})
        L.append(f'| {s} | {a[0]} | {acc_a:.2f} | {acc_b:.2f} | {acc_b - acc_a:+.2f} | {a[2]} | {b[2]} |')

    os.makedirs(args.out, exist_ok=True)
    md = '\n'.join(L) + '\n'
    with open(os.path.join(args.out, 'compare.md'), 'w') as f:
        f.write(md)
    strip = lambda s: {k: v for k, v in s.items() if k != 'subjects'}  # noqa: E731
    with open(os.path.join(args.out, 'compare.json'), 'w') as f:
        json.dump({'scores': args.scores, na: strip(sa), nb: strip(sb), 'paired': paired, 'subjects': per}, f, indent=2)
    with open(os.path.join(args.out, 'compare_per_question.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'subject', 'answer', f'parsed_{na}', f'parsed_{nb}', f'correct_{na}', f'correct_{nb}',
                    f'finish_{na}', f'finish_{nb}'])
        for i in ids:
            w.writerow([i, A[i]['subject'], A[i]['answer'], A[i]['parsed'], B[i]['parsed'], A[i]['correct'],
                        B[i]['correct'], A[i]['finish_reason'], B[i]['finish_reason']])
    print(md)


if __name__ == '__main__':
    main()
