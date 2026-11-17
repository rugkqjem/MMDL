"""Build DPO preference pairs from the rollouts (correct pairs, cut loop pairs) and dpo_resample.py output. CPU only.

Questions: data/rl/train_final.jsonl only (already decontaminated, >= 3 options, format/label exclusions applied;
the exclusion lists are asserted again). Every pair keeps the question's eval prompt and images.

correct pair  chosen   = finished (EOS) response scored correct by the eval scorer
              rejected = finished response, answer parsed by rule (no random fallback), scored wrong
              lengths within --len_ratio of each other (DPO otherwise learns "shorter/longer wins"); the closest
              pair in length is taken, at most --correct_per_q per question
loop pair, --loop_mode resample (needs dpo_resample.py output)
              prefix   = original looping response up to the kept branch point (shared, goes into the prompt, so
                         its tokens cancel in the DPO loss; only what comes after the branch is trained)
              chosen   = an escaping continuation from that prefix (ends with EOS); correct ones preferred,
                         --loop_chosen_correct keeps only correct ones
              rejected = the original continuation from the same prefix, NO EOS, at least as long as chosen and
                         --min_rejected chars, extended until >= --min_loop_share of it is repeat/restart text
                         (loop_onset.unit_flags) so it shows the loop
              at most --loop_per_q per question
loop pair, --loop_mode cut (Seunghee's plan "SFT -> DPO 계획 초안", no GPU needed)
              no shared prefix
              chosen   = another finished, correct response to the same question; "mismatch then recovered"
                         responses (said the result is not among the options, still answered correctly) first
              rejected = a truncated response that says the result is not among the options (MISMATCH), NO EOS,
                         cut at max(chosen tokens, first mismatch + --cut_extra_tokens), at most --cut_max_tokens,
                         on a line break. Tokens are estimated per response from its own chars/token (8192 tokens);
                         responses whose first mismatch is past the cap are skipped
              --cut_scope all: any truncated response; the cut then uses chosen tokens and --cut_extra_tokens

  python scripts/dpo_pairs.py                                        # correct pairs only
  python scripts/dpo_pairs.py --loop_mode cut                        # + loop pairs, Seunghee's plan (now)
  python scripts/dpo_pairs.py --resample outputs/dpo/resample.jsonl  # + loop pairs from resampling (after RFT)
Output: outputs/dpo/pairs.jsonl (+ pairs_report.md). Rows: id, kind, prompt, images, options, answer, prefix,
chosen, rejected, rejected_eos, meta.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_onset import loop_share, unit_flags  # noqa: E402
from rl_reward import score_response  # noqa: E402

MISMATCH = re.compile(
    r"(not (in|among|one of|listed (in|among)|present in) (the )?(given |answer |provided |listed )?(options|choices))"
    r"|(none of the (given |answer |provided )?(options|choices))"
    r"|((doesn[’']t|does not|don[’']t|do not) (match|correspond to) (any (of the )?|the |one of the )?"
    r"(given |answer )?(options?|choices?))"
    r"|(no (option|choice) (matches|is|corresponds))|(not an? (option|choice))"
    r"|(isn[’']t (an? |one of the )?(options?|choices?))", re.I)


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
    # an escape must write something and end with a parsable answer: an immediate EOS right after the prefix
    # (empty text) is not a usable chosen side
    esc = [c for c in rs['continuations'] if c['finish_reason'] == 'stop' and c['text'].strip() and c.get('parsed')]
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


def first_mismatch(text):
    m = MISMATCH.search(text)
    return m.start() if m else None


def cut_on_line(text, n):
    """text[:n] ending at a line break (else at a space) so the rejected side ends cleanly."""
    if n >= len(text):
        return text
    k = text.rfind('\n', 0, n)
    return text[:k + 1] if k > n * 0.8 else cut_at_space(text, n)


def cut_loop_pairs(q, samples, per_q, scope, extra_tokens, max_tokens, flags_of=unit_flags):
    """Team plan loop pairs for one question. Returns (pairs, skipped_reasons)."""
    good = [s for s in samples if s['finish_reason'] == 'stop'
            and score_response(s['response'], q['answer'], q['options'], False)['correct']]
    if not good:
        return [], ['no correct']
    rec = [s for s in good if first_mismatch(s['response']) is not None]
    order = rec + [s for s in good if s not in rec]
    loops, skipped = [], []
    for s in samples:
        if s['finish_reason'] != 'length':
            continue
        mm = first_mismatch(s['response'])
        if scope == 'mismatch' and mm is None:
            continue
        loops.append((s, mm))
    out = []
    for i, (s, mm) in enumerate(loops):
        if len(out) == per_q:
            break
        ch = order[len(out) % len(order)]
        text = s['response']
        cpt = len(text) / max(s['num_output_tokens'], 1)                 # chars per token of this response
        mm_tok = mm / cpt if mm is not None else 0
        if mm is not None and mm_tok >= max_tokens:
            skipped.append('mismatch past cap')
            continue
        tok = min(max(ch['num_output_tokens'], mm_tok + extra_tokens), max_tokens)
        rej = cut_on_line(text, int(tok * cpt))
        if mm is not None and len(rej) <= mm:
            rej = cut_on_line(text, int(min(mm_tok + extra_tokens, max_tokens) * cpt))
        fl = flags_of(text)
        out.append({'kind': 'loop', 'prefix': '', 'chosen': ch['response'], 'rejected': rej, 'rejected_eos': False,
                    'meta': {'mode': 'cut', 'sample': s['_k'], 'chosen_sample': ch['_k'],
                             'chosen_recovered': ch in rec, 'mismatch_char': mm,
                             'chosen_tokens': ch['num_output_tokens'], 'rejected_tokens_est': round(len(rej) / cpt),
                             'chosen_chars': len(ch['response']), 'rejected_chars': len(rej),
                             'rejected_before_mismatch': round((mm if mm is not None else 0) / len(rej), 3),
                             'rejected_loop_share': round(loop_share(fl, 0, len(rej)), 3),
                             'rejected_to_end': len(rej) >= len(text) - 1}})
    return out, skipped


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
        elif p['meta'].get('mode') == 'cut':
            m = p['meta']
            assert not p['rejected_eos'] and p['prefix'] == ''
            assert score_response(p['chosen'], p['answer'], p['options'], False)['correct']
            assert m['mismatch_char'] is None or m['mismatch_char'] < len(p['rejected']), 'mismatch not in rejected'
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
    p.add_argument('--loop_mode', choices=['resample', 'cut'], default=None,
                   help='resample: from --resample output; cut: Seunghee\'s plan from the rollouts (default: resample if '
                        '--resample is given, else no loop pairs)')
    p.add_argument('--resample', default=None, help='dpo_resample.py output (loop_mode resample)')
    p.add_argument('--cut_scope', choices=['mismatch', 'all'], default='mismatch',
                   help='cut mode: only truncated responses that say the result is not among the options (core)')
    p.add_argument('--cut_extra_tokens', type=int, default=1500)
    p.add_argument('--cut_max_tokens', type=int, default=4095)
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
    if a.loop_mode is None and a.resample:
        a.loop_mode = 'resample'
    if a.loop_mode == 'resample' and not a.resample:
        p.error('--loop_mode resample needs --resample')
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
    skipped = Counter()
    if a.loop_mode == 'cut':
        for qid in sorted(samples):
            prs, sk = cut_loop_pairs(qs[qid], samples[qid], a.loop_per_q, a.cut_scope, a.cut_extra_tokens,
                                     a.cut_max_tokens)
            skipped.update(r for r in sk if r != 'no correct')
            pairs.extend(pr | {'id': qid} for pr in prs)
    if a.loop_mode == 'resample':
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
    if a.loop_mode == 'cut':
        lp = [p['meta'] for p in pairs if p['kind'] == 'loop']
        lines.append(f'- loop pairs (cut, scope {a.cut_scope}): {kinds["loop"]} from '
                     f'{len({p["id"] for p in pairs if p["kind"] == "loop"})} questions; skipped {dict(skipped)}')
        if lp:
            lines.append(f'  - chosen "mismatch then recovered" {sum(m["chosen_recovered"] for m in lp)}/{len(lp)}; '
                         f'chosen tokens median {median(m["chosen_tokens"] for m in lp):.0f}, rejected (est.) '
                         f'{median(m["rejected_tokens_est"] for m in lp):.0f}')
            lines.append(f'  - rejected: share before the first mismatch median '
                         f'{median(m["rejected_before_mismatch"] for m in lp):.2f}, repeat/restart text share median '
                         f'{median(m["rejected_loop_share"] for m in lp):.2f}')
    if a.loop_mode == 'resample':
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
