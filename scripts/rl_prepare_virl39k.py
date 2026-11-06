"""ViRL39K (local HF snapshot) -> qwen-vl-finetune JSONL for decontaminate.py / rl_rollout_filter.py.

The user turn is built by infer.build_messages(), so the prompt is byte-identical to the MMMU eval prompt
(same as prepare_mmmu_sft.py). Rows also carry the HF-MMMU-like fields (question / options / answer /
question_type) that infer.prepare_sample() and score.score_qwen() need, so rollouts and RL rewards are scored
exactly like the eval. This script only reformats the *questions*; the answer parser (score.py) is untouched.

ViRL39K inlines the options in the question text in several layouts, e.g.
    "... Choices: (A) yes (B) no"          "...\\nOptions:\\nA. 3\\nB. 4"         "...\\nA: 3\\nB: 4\\nC: 5"
    "... (A) 30° (B) 45° (C) 60°"          "... Choices: A. 3 B. 4 C. 5"
parse_question() finds an A, B, C, ... marker sequence, splits it off as the options list and keeps the rest as
the stem. Gold answers are letters ("\\boxed{B}", "(B)", "B. 45°") or the text of one option; both are mapped to
the option letter. Questions without a parsable option list stay out (open-ended by default, see below).

Multiple-choice only by default. For open questions score_qwen() turns the gold into choice A of
{A: gold, B: 'Other Answers'} and can_infer_text() accepts any prediction that *contains* the gold string
("13" matches gold "3"). That is fine for reporting a fixed model but is a reward hack waiting to happen in RL.
MMMU-val is ~94% multiple-choice anyway. --keep_open includes them.

Download first:
  hf download TIGER-Lab/ViRL39K --repo-type dataset --local-dir data/virl39k_raw
Usage (from the repo root):
  python scripts/rl_prepare_virl39k.py --src data/virl39k_raw --out data/virl39k/all.jsonl --dump_skipped 15
  python scripts/rl_prepare_virl39k.py --src data/virl39k_raw --out data/virl39k/sub.jsonl --max_per_category 3000
Images stay in the snapshot: pass --src as --image_root to the next scripts.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import string
import sys
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import build_messages  # noqa: E402

LETTERS = string.ascii_uppercase[:10]  # A-J
IMG_TAG = re.compile(r'<image\s*\d*>', re.I)
# trailing header right before the options ("Choices:", "Options:", "Answer choices:", Chinese "选项")
HDR_TAIL = re.compile(r'(?:\b(?:answer\s+)?(?:choices|options)\b|\b(?:select|choose) from\b|选项)\s*[:：]?\s*$', re.I)
# instruction lines that belong to the source dataset's prompt, not to the question
INSTR = re.compile(r"option'?s? letter|correct option|answer with the|answer the question using|"
                   r'choose the correct|select the correct|please answer|from the given choices', re.I)
# marker styles for letter L, most reliable first
STYLES = {
    'paren': lambda L: re.compile(r'\(' + L + r'\)'),                                  # (A)
    'line': lambda L: re.compile(r'(?:^|\n)[ \t]*\(?' + L + r'\s*[.:：),]'),            # A.  A)  A:  A,  at line start
    'inline': lambda L: re.compile(r'(?:(?<=\s)|(?<=^))' + L + r'[.:)]\s'),           # " A. " anywhere
}


def strip_boxed(ans):
    """'\\boxed{A}' -> 'A' (last boxed, braces balanced); unchanged if there is none."""
    s = str(ans).strip()
    i = s.rfind('\\boxed{')
    if i < 0:
        return s
    depth, j0 = 0, i + len('\\boxed')
    for j in range(j0, len(s)):
        depth += {'{': 1, '}': -1}.get(s[j], 0)
        if depth == 0:
            return s[j0 + 1:j].strip()
    return s


def _find_sequence(text, style):
    """Last 'A' marker that is followed by B, C, ... in order -> [(letter, start, end)] (>= 2 letters) or None."""
    pat = {L: STYLES[style](L) for L in LETTERS}
    for a in reversed(list(pat['A'].finditer(text))):
        seq, pos = [('A', a.start(), a.end())], a.end()
        for L in LETTERS[1:]:
            m = pat[L].search(text, pos)
            if not m:
                break
            seq.append((L, m.start(), m.end()))
            pos = m.end()
        if len(seq) >= 2:
            return seq
    return None


def drop_instructions(line):
    """Remove only the instruction *sentences* of a line ('Please select the correct options:'), keep the rest."""
    parts = re.split(r'(?<=[.?!:：])\s+', line.strip())
    keep = [x for x in parts if x and not INSTR.search(x) and not re.fullmatch(r'(?:hint|note)\s*[:：]', x, re.I)]
    return ' '.join(keep).strip()


def _clean_opt(t):
    t = t.strip().rstrip(',;').strip()
    return re.sub(r'\s+', ' ', t)


def split_options(q):
    """-> (stem, options) or (q, []) if no option list is found."""
    for style in ('paren', 'line', 'inline'):
        seq = _find_sequence(q, style)
        if not seq:
            continue
        if style == 'inline' and not HDR_TAIL.search(q[:seq[0][1]].rstrip()):
            continue  # bare " A. " is too ambiguous without a "Choices:" header right before it
        opts, tail = [], ''
        for i, (_, _, end) in enumerate(seq):
            nxt = seq[i + 1][1] if i + 1 < len(seq) else len(q)
            chunk = q[end:nxt]
            if i + 1 == len(seq):  # last option ends at the line break; anything after is trailing text
                first, _, tail = chunk.strip().partition('\n')
                chunk = first
            opts.append(_clean_opt(chunk))
        if any(not o or len(o) > 400 for o in opts):
            continue
        stem = HDR_TAIL.sub('', q[:seq[0][1]].rstrip()).rstrip()
        extra = [drop_instructions(ln) for ln in tail.split('\n')]
        stem = '\n'.join([stem] + [x for x in extra if x]).strip()
        stem = '\n'.join(x for x in (drop_instructions(ln) for ln in stem.split('\n')) if x).strip()
        stem = re.sub(r'^\s*question\s*[:：]\s*', '', stem, flags=re.I | re.M)  # build_messages adds 'Question: '
        if stem:
            return stem, opts
    return q, []


_NORM = lambda s: re.sub(r'[\s$]+', '', re.sub(r'\\text\{([^{}]*)\}', r'\1', str(s))).lower().rstrip('.')


def gold_letter(gold, options, text_only=False):
    """Letter gold ('B', '(B)', 'B. 45°', '\\text{B}') or the exact text of one option -> 'B'; else None.
    text_only: the options carried no letters in the source, so the gold is option *text* (it can itself be a
    letter, e.g. options 'C / A / D / B' with gold 'D' means the third option, not the fourth)."""
    g = re.sub(r'\\text\{([^{}]*)\}', r'\1', gold).strip()
    hits = [LETTERS[i] for i, o in enumerate(options) if _NORM(o) == _NORM(g)]
    if text_only:
        return hits[0] if len(hits) == 1 else None
    m = re.fullmatch(r'\(?([A-J])\)?(?:[.:)]\s*.*)?', g, flags=re.S)
    if m and m.group(1) in LETTERS[:len(options)]:
        return m.group(1)
    return hits[0] if len(hits) == 1 else None


# "<question>?\nJ\nG\nA\nE\nPlease answer the question based on the options mentioned before." (AI2D-style)
UNLETTERED_INSTR = re.compile(r'options (?:mentioned before|above|given above)|(?:from|among) the (?:following|options)',
                              re.I)


def split_unlettered(q):
    """Options given one per line without letters, closed by an instruction line -> (stem, options) or (q, [])."""
    lines = [ln.strip() for ln in q.split('\n')]
    k = next((i for i in range(len(lines) - 1, -1, -1) if UNLETTERED_INSTR.search(lines[i])), None)
    if k is None:
        return q, []
    j = k - 1
    while j >= 0 and lines[j] and len(lines[j]) <= 120 and not lines[j].endswith(('?', ':', '：')):
        j -= 1
    opts = [ln for ln in lines[j + 1:k] if ln]
    stem = '\n'.join(ln for ln in lines[:j + 1] if ln).strip()
    if not (2 <= len(opts) <= len(LETTERS)) or not stem or len(set(map(_NORM, opts))) != len(opts):
        return q, []
    stem = re.sub(r'^\s*question\s*[:：]\s*', '', stem, flags=re.I)
    return stem, opts


def normalize_text(raw_q):
    """Drop image tags; turn literal '\\n' (and '/n' left between image tags) into real line breaks."""
    q = IMG_TAG.sub('', raw_q)
    q = re.sub(r'^(?:\s*/n)+', '', q)              # '<image>/n<image>/nBased on ...'
    q = re.sub(r'\\n(?![a-z])', '\n', q)          # literal backslash-n, but not \\nu, \\neq, \\nabla, ...
    return q.strip()


def parse_question(raw_q, gold):
    """-> (stem, options, letter_or_None, kind) with kind in {mc, mc_gold_not_in_options, mc_unparsed, open}."""
    q = normalize_text(raw_q)
    stem, opts = split_options(q)
    text_only = False
    if not opts:
        stem, opts = split_unlettered(q)
        text_only = bool(opts)
    if opts:
        letter = gold_letter(gold, opts, text_only)
        return stem, opts, letter, ('mc' if letter else 'mc_gold_not_in_options')
    if re.fullmatch(r'\(?[A-J]\)?\.?', gold.strip()):
        return q, [], None, 'mc_unparsed'
    return q, [], None, 'open'


def load_table(src):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(src, '**', '*.parquet'), recursive=True))
    if not files:
        sys.exit(f'no parquet files under {src}')
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def ensure_images(src):
    img_dir = os.path.join(src, 'images')
    if os.path.isdir(img_dir) and any(os.scandir(img_dir)):
        return
    for z in glob.glob(os.path.join(src, '**', '*.zip'), recursive=True):
        print(f'[img] extracting {z}', file=sys.stderr)
        with zipfile.ZipFile(z) as f:
            f.extractall(src)


def passrate(v):
    """PassRate_32BTrained; 0.0 is a real value (never solved), only missing/NaN becomes -1."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return -1.0
    return -1.0 if math.isnan(v) else v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='local snapshot of TIGER-Lab/ViRL39K')
    p.add_argument('--out', required=True)
    p.add_argument('--include_categories', nargs='*', default=None, help='substring match')
    p.add_argument('--exclude_categories', nargs='*', default=None, help='substring match, e.g. GradeSchool')
    p.add_argument('--max_per_category', type=int, default=None)
    p.add_argument('--max_images', type=int, default=7, help='infer.make_llm allows 7 images per prompt')
    p.add_argument('--keep_open', action='store_true', help='also keep open-ended questions (see docstring)')
    p.add_argument('--dump_skipped', type=int, default=0,
                   help='write N random examples per skip reason to <out>_skipped.txt (to check the parser)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    ensure_images(args.src)
    df = load_table(args.src)
    print(f'[load] {len(df)} rows, columns={list(df.columns)}', file=sys.stderr)

    rows, skipped, examples = [], Counter(), defaultdict(list)
    kinds_by_cat = defaultdict(Counter)
    for r in df.to_dict('records'):
        cat = str(r.get('category', ''))
        if args.include_categories and not any(c.lower() in cat.lower() for c in args.include_categories):
            skipped['category'] += 1; continue
        if args.exclude_categories and any(c.lower() in cat.lower() for c in args.exclude_categories):
            skipped['category'] += 1; continue
        imgs = r.get('image')
        imgs = [imgs] if isinstance(imgs, str) else list(imgs if imgs is not None else [])
        if not imgs:
            skipped['no_image'] += 1; continue
        if len(imgs) > args.max_images:
            skipped['too_many_images'] += 1; continue
        if any(not os.path.exists(os.path.join(args.src, i)) for i in imgs):
            skipped['missing_image_file'] += 1; continue
        gold = strip_boxed(r['answer'])
        stem, opts, letter, kind = parse_question(str(r['question']), gold)
        kinds_by_cat[cat][kind] += 1
        if kind != 'mc' and not (kind == 'open' and args.keep_open):
            skipped[kind] += 1
            examples[kind].append((cat, str(r['question']), gold))
            continue
        answer = letter if kind == 'mc' else gold
        sample = {'question': stem, 'options': repr(opts) if opts else '',
                  **{f'image_{i + 1}': path for i, path in enumerate(imgs)}}
        _, prompt, _ = build_messages(sample, None, None)  # image entries are placeholders; only text is used
        rows.append({
            'id': str(r.get('qid')), 'image': imgs,
            'conversations': [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': f'Answer: {answer}'}],
            'question': stem, 'options': opts, 'answer': answer,
            'question_type': 'multiple-choice' if opts else 'open',
            'category': cat, 'source': str(r.get('source', '')),
            'passrate_32b': passrate(r.get('PassRate_32BTrained')),
        })

    if args.max_per_category:
        rng, by = random.Random(args.seed), defaultdict(list)
        for x in rows:
            by[x['category']].append(x)
        rows = []
        for _, xs in sorted(by.items()):
            rng.shuffle(xs)
            rows += xs[:args.max_per_category]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')

    if args.dump_skipped:
        rng = random.Random(args.seed)
        path = os.path.splitext(args.out)[0] + '_skipped.txt'
        with open(path, 'w', encoding='utf-8') as f:
            for kind, xs in sorted(examples.items()):
                f.write(f'\n######## {kind}: {len(xs)}\n')
                f.write(f'categories: {Counter(c for c, _, _ in xs).most_common()}\n')
                for c, q, g in rng.sample(xs, min(args.dump_skipped, len(xs))):
                    f.write(f'--- [{c}] gold={g!r}\n{q[-500:]}\n')
        print(f'[dump] skipped examples -> {path}')

    print(f'[done] {len(rows)} rows -> {args.out}')
    print(f'  question_type: {dict(Counter(x["question_type"] for x in rows))}')
    print(f'  skipped: {dict(skipped)}')
    print('  per category (kept / mc / mc_gold_not_in_options / mc_unparsed / open):')
    kept = Counter(x['category'] for x in rows)
    for c, k in sorted(kinds_by_cat.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"    {kept[c]:6d} / {k['mc']:6d} / {k['mc_gold_not_in_options']:5d} / {k['mc_unparsed']:5d} / "
              f"{k['open']:6d}  {c}")


if __name__ == '__main__':
    main()
