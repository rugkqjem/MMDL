"""ViRL39K (local HF snapshot) -> qwen-vl-finetune JSONL for decontaminate.py / rl_rollout_filter.py.

The user turn is built by infer.build_messages(), so the prompt is byte-identical to the MMMU eval prompt
(same as prepare_mmmu_sft.py). Rows also carry the HF-MMMU-like fields (question / options / answer /
question_type) that infer.prepare_sample() and score.score_qwen() need, so rollouts and RL rewards are scored
exactly like the eval.

Multiple-choice only by default. For open questions score_qwen() turns the gold into choice A of
{A: gold, B: 'Other Answers'} and can_infer_text() accepts any prediction that *contains* the gold string
("13" matches gold "3"). That is fine for reporting a fixed model but is a reward hack waiting to happen in RL.
MMMU-val is ~94% multiple-choice anyway. --keep_open includes them.

Download first:
  huggingface-cli download TIGER-Lab/ViRL39K --repo-type dataset --local-dir data/virl39k_raw
Usage (from the repo root):
  python scripts/rl_prepare_virl39k.py --src data/virl39k_raw --out data/virl39k/all.jsonl
  python scripts/rl_prepare_virl39k.py --src data/virl39k_raw --out data/virl39k/sub.jsonl \\
      --exclude_categories GradeSchool --max_per_category 3000
Images stay in the snapshot: pass --src as --image_root to the next scripts.
"""
import argparse
import glob
import os
import random
import re
import string
import sys
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json  # noqa: E402

from infer import build_messages  # noqa: E402

IMG_TAG = re.compile(r'<image\s*\d*>', re.I)
CHOICE_HDR = re.compile(r'(?:^|\s)(?:Choices|Options)\s*:\s*', re.I)
PAREN_OPT = re.compile(r'\(([A-J])\)\s*')
LINE_OPT = re.compile(r'(?:^|\n)\s*([A-J])[.:)]\s+')


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


def _split_options(tail, pattern):
    parts = pattern.split(tail)  # [pre, 'A', textA, 'B', textB, ...]
    if len(parts) < 5:
        return None
    letters, texts = parts[1::2], [t.strip().rstrip(',;').strip() for t in parts[2::2]]
    if letters != list(string.ascii_uppercase[:len(letters)]) or any(not t for t in texts):
        return None
    return texts


def parse_question(raw_q, gold):
    """-> (stem, options, kind) with kind in {mc, open, mc_unparsed}. ViRL inlines choices as 'Choices: (A) .. (B) ..'."""
    q = IMG_TAG.sub('', raw_q).strip()
    m = None
    for m in CHOICE_HDR.finditer(q):
        pass  # last header wins
    if m:
        opts = _split_options(q[m.end():], PAREN_OPT) or _split_options('\n' + q[m.end():], LINE_OPT)
        if opts:
            return q[:m.start()].strip(), opts, 'mc'
    if re.fullmatch(r'\(?[A-J]\)?\.?', gold):
        return q, [], 'mc_unparsed'  # letter gold, options not parsable -> score_qwen cannot grade it
    return q, [], 'open'


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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='local snapshot of TIGER-Lab/ViRL39K')
    p.add_argument('--out', required=True)
    p.add_argument('--include_categories', nargs='*', default=None, help='substring match')
    p.add_argument('--exclude_categories', nargs='*', default=None, help='substring match, e.g. GradeSchool')
    p.add_argument('--max_per_category', type=int, default=None)
    p.add_argument('--max_images', type=int, default=7, help='infer.make_llm allows 7 images per prompt')
    p.add_argument('--keep_open', action='store_true', help='also keep open-ended questions (see docstring)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    ensure_images(args.src)
    df = load_table(args.src)
    print(f'[load] {len(df)} rows, columns={list(df.columns)}', file=sys.stderr)

    rows, skipped = [], Counter()
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
        stem, opts, kind = parse_question(str(r['question']), gold)
        if kind == 'mc_unparsed':
            skipped['mc_options_unparsed'] += 1; continue
        if kind == 'open' and not args.keep_open:
            skipped['open_ended'] += 1; continue
        if kind == 'mc':
            gold = gold.strip('().').upper()
            if gold not in string.ascii_uppercase[:len(opts)]:
                skipped['answer_not_in_options'] += 1; continue
        # same prompt builder as the eval; image entries are placeholders here (only the text is used)
        sample = {'question': stem, 'options': repr(opts) if opts else '',
                  **{f'image_{i + 1}': path for i, path in enumerate(imgs)}}
        _, prompt, _ = build_messages(sample, None, None)
        rows.append({
            'id': str(r.get('qid')), 'image': imgs,
            'conversations': [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': f'Answer: {gold}'}],
            'question': stem, 'options': opts, 'answer': gold,
            'question_type': 'multiple-choice' if opts else 'open',
            'category': cat, 'source': str(r.get('source', '')),
            'passrate_32b': float(r.get('PassRate_32BTrained', -1) or -1),
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
    print(f'[done] {len(rows)} rows -> {args.out}')
    print(f'  question_type: {dict(Counter(x["question_type"] for x in rows))}')
    print(f'  skipped: {dict(skipped)}')
    print('  categories:')
    for c, n in Counter(x['category'] for x in rows).most_common():
        print(f'    {n:6d}  {c}')


if __name__ == '__main__':
    main()
