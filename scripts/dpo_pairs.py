"""Build DPO preference pairs from the rollouts (correct pairs) and dpo_resample.py output (loop pairs). CPU only.

Questions: data/rl/train_final.jsonl only (already decontaminated, >= 3 options, format/label exclusions applied;
the exclusion lists are asserted again). Every pair keeps the question's eval prompt and images.

correct pair  chosen   = finished (EOS) response scored correct by the eval scorer
              rejected = finished response, answer parsed by rule (no random fallback), scored wrong
              lengths within --len_ratio of each other (DPO otherwise learns "shorter/longer wins"); the closest
              pair in length is taken, at most --correct_per_q per question
loop pair     prefix   = original looping response up to the kept branch point (shared, goes into the prompt, so
                         its tokens cancel in the DPO loss; only what comes after the branch is trained)
              chosen   = an escaping continuation from that prefix (ends with EOS); correct ones preferred,
                         --loop_chosen_correct keeps only correct ones
              rejected = the original continuation from the same prefix, NO EOS, at least as long as chosen and
                         --min_rejected chars, extended until >= --min_loop_share of it is repeat/restart text
                         (loop_onset.unit_flags) so it shows the loop
              at most --loop_per_q per question

  python scripts/dpo_pairs.py                                        # correct pairs only (no resample yet)
  python scripts/dpo_pairs.py --resample outputs/dpo/resample.jsonl  # + loop pairs
Output: outputs/dpo/pairs.jsonl (+ pairs_report.md). Rows: id, kind, prompt, images, options, answer, prefix,
chosen, rejected, rejected_eos, meta.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_onset import loop_share, unit_flags  # noqa: E402
from rl_reward import score_response  # noqa: E402


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def read_ids(path):
    with open(path, encoding='utf-8') as f:
        return {line.split('\t')[0].strip() for line in f if line.strip() and not line.startswith('#')}


def cut_at_space(text, n):
    """text[:n], moved back to the last whitespace so no word is split."""
    if n >= len(text):
        return text
    k = text.rfind(' ', 0, n)
    k2 = text.rfind('\n', 0, n)
    k = max(k, k2)
    return text[:k] if k > n * 0.8 else text[:n]


def correct_pairs(q, samples, len_ratio, per_q):
    good, bad = [], []
    for s in samples:
        if s['finish_reason'] != 'stop':
            continue
        sc = score_response(s['response'], q['answer'], q['options'], truncated=False)
        if sc['correct']:
            good.append(s)
        elif sc['method'] == 'rule':          # parsed by rule: a real wrong answer, not a parse failure
            bad.append(s)
    cands = []
    for g in good:
        for b in bad:
            lg, lb = g['num_output_tokens'], b['num_output_tokens']
            r = max(lg, lb) / max(min(lg, lb), 1)
            if r <= len_ratio:
                cands.append((r, g, b))
    cands.sort(key=lambda x: x[0])
    out, used = [], set()
    for r, g, b in cands:
        if id(g) in used or id(b) in used:
            continue
        used.update((id(g), id(b)))
        out.append({'kind': 'correct', 'prefix': '', 'chosen': g['response'], 'rejected': b['response'],
                    'rejected_eos': True,
                    'meta': {'chosen_tokens': g['num_output_tokens'], 'rejected_tokens': b['num_output_tokens'],
                             'len_ratio': round(r, 3), 'chosen_sample': g['_k'], 'rejected_sample': b['_k']}})
        if len(out) == per_q:
            break
    return out


def loop_pair(q, rs, original, rng, chosen_correct, min_rejected, min_share, flags=None):
    """Shared prefix = original[:branch]; chosen = an escaping continuation; rejected = what the original did next,
    at least as long as chosen and min_rejected chars, extended until >= min_share of it is repeat/restart text
    (or the original ends), so the rejected side shows the loop and not only ordinary reasoning."""
    if rs['branch_char'] is None:
        return None
    b = rs['branch_char']
    esc = [c for c in rs['continuations'] if c['finish_reason'] == 'stop']
    pool = [c for c in esc if c['correct']] or ([] if chosen_correct else esc)
    if not pool:
        return None
    ch = rng.choice(pool)
    fl = flags or unit_flags(original)
    n = max(len(ch['text']), min_rejected)
    while b + n < len(original) and loop_share(fl, b, b + n) < min_share:
        n += 1000
    rej = cut_at_space(original[b:], n)
    share = loop_share(fl, b, b + len(rej))
    return {'kind': 'loop', 'prefix': original[:b], 'chosen': ch['text'], 'rejected': rej, 'rejected_eos': False,
            'meta': {'sample': rs['sample'], 'branch_char': b, 'branch_frac': round(b / len(original), 2),
                     'escaped': sum(c['finish_reason'] == 'stop' for c in rs['continuations']),
                     'n': len(rs['continuations']), 'chosen_correct': ch['correct'],
                     'chosen_chars': len(ch['text']), 'rejected_chars': len(rej),
                     'rejected_loop_share': round(share, 3), 'rejected_to_end': b + len(rej) >= len(original) - 1}}


def check(pairs, keep, excluded, per_q_limits, min_share=0.5):
    per = Counter((p['id'], p['kind']) for p in pairs)
    for p in pairs:
        assert p['id'] in keep and p['id'] not in excluded, p['id']
        assert p['chosen'].strip() and p['rejected'].strip(), p['id']
        assert '<|im_end|>' not in p['chosen'] + p['rejected'] + p['prefix'], 'stop token inside text'
        if p['kind'] == 'correct':
            assert p['prefix'] == '' and p['rejected_eos']
            assert score_response(p['chosen'], p['answer'], p['options'], False)['correct']
            r = score_response(p['rejected'], p['answer'], p['options'], False)
            assert not r['correct'] and r['method'] == 'rule'
        else:
            assert not p['rejected_eos'] and p['prefix']
            m = p['meta']
            assert m['rejected_loop_share'] >= min_share or m['rejected_to_end'], 'rejected has no loop text'
    for (q, kind), n in per.items():
        assert n <= per_q_limits[kind], (q, kind, n)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--train', default='data/rl/train_final.jsonl')
    p.add_argument('--resp', nargs='+', default=['outputs/rl_rollout/rollouts_resp.jsonl',
                                                 'outputs/rl_rollout/pilot_resp.jsonl'])
    p.add_argument('--exclude', nargs='*', default=['data/rl/exclude_format_only.txt',
                                                    'data/rl/exclude_label_issues.txt'])
    p.add_argument('--resample', default=None, help='dpo_resample.py output; omit for correct pairs only')
    p.add_argument('--out', default='outputs/dpo/pairs.jsonl')
    p.add_argument('--no_correct', action='store_true', help='loop pairs only')
    p.add_argument('--len_ratio', type=float, default=1.5)
    p.add_argument('--correct_per_q', type=int, default=1)
    p.add_argument('--loop_per_q', type=int, default=2)
    p.add_argument('--loop_chosen_correct', action='store_true', help='chosen must also be correct')
    p.add_argument('--min_rejected', type=int, default=2000, help='rejected chars at least (and >= chosen)')
    p.add_argument('--min_loop_share', type=float, default=0.5,
                   help='rejected is extended until this share of it is repeat/restart text (loop_onset)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)

    qs = {r['id']: r for r in read_jsonl(a.train)}
    excluded = set().union(*(read_ids(f) for f in a.exclude if os.path.exists(f)))
    samples = defaultdict(list)
    seen = set()
    for path in a.resp:
        for g in read_jsonl(path):
            if g['id'] not in qs or g['id'] in seen:
                continue
            seen.add(g['id'])
            for k, s in enumerate(g['samples']):
                samples[g['id']].append(dict(s, _k=k))

    pairs = []
    if not a.no_correct:
        for qid in sorted(samples):
            for pr in correct_pairs(qs[qid], samples[qid], a.len_ratio, a.correct_per_q):
                pairs.append(pr | {'id': qid})
    n_loop_in = n_loop_none = 0
    if a.resample:
        by_q = defaultdict(list)
        for rs in read_jsonl(a.resample):
            by_q[rs['id']].append(rs)
        for qid in sorted(by_q):
            made = 0
            for rs in sorted(by_q[qid], key=lambda r: r['sample']):
                n_loop_in += 1
                orig = samples[qid][rs['sample']]['response'] if qid in samples else None
                pr = loop_pair(qs[qid], rs, orig, rng, a.loop_chosen_correct, a.min_rejected, a.min_loop_share) if orig else None
                if pr is None:
                    n_loop_none += 1
                    continue
                pairs.append(pr | {'id': qid})
                made += 1
                if made == a.loop_per_q:
                    break
    for pr in pairs:
        q = qs[pr['id']]
        pr.update(prompt=q['prompt'], images=q['images'], options=q['options'], answer=q['answer'],
                  category=q.get('category', ''))
    check(pairs, set(qs), excluded, {'correct': a.correct_per_q, 'loop': a.loop_per_q}, a.min_loop_share)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        for pr in pairs:
            f.write(json.dumps(pr, ensure_ascii=False) + '\n')
    kinds = Counter(p['kind'] for p in pairs)
    lines = [f'# DPO pairs ({len(pairs)})', '', f'- args: `{json.dumps(vars(a), ensure_ascii=False)}`',
             f'- correct pairs: {kinds["correct"]} from {len({p["id"] for p in pairs if p["kind"] == "correct"})} '
             f'questions (of {len(samples)} with rollouts)']
    cp = [p['meta'] for p in pairs if p['kind'] == 'correct']
    if cp:
        lines.append(f'  - length ratio median {median(m["len_ratio"] for m in cp):.2f}; chosen tokens median '
                     f'{median(m["chosen_tokens"] for m in cp):.0f}, rejected {median(m["rejected_tokens"] for m in cp):.0f}')
    if a.resample:
        lp = [p['meta'] for p in pairs if p['kind'] == 'loop']
        lines.append(f'- loop pairs: {kinds["loop"]} (resampled responses {n_loop_in}, no usable escape {n_loop_none})')
        if lp:
            lines.append(f'  - branch at (fraction of response) {dict(Counter(m["branch_frac"] for m in lp))}; '
                         f'rejected loop share median {median(m["rejected_loop_share"] for m in lp):.2f}')
            lines.append(f'  - chosen correct {sum(m["chosen_correct"] for m in lp)}/{len(lp)}; escape rate at kept '
                         f'branch median {median(m["escaped"] / m["n"] for m in lp):.2f}; chosen chars median '
                         f'{median(m["chosen_chars"] for m in lp):.0f}, rejected {median(m["rejected_chars"] for m in lp):.0f}')
    lines.append('- categories: ' + ', '.join(f'{k} {v}' for k, v in Counter(p['category'] for p in pairs).most_common()))
    with open(os.path.splitext(a.out)[0] + '_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
