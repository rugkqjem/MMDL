"""Classify run-1 wrong answers and run-1 -> run-2 correctness flips (CPU only, no re-inference).

  python scripts/classify_errors.py \
      --run1 outputs/qwen3vl4b_mmmu_val --run2 outputs/qwen3vl4b_mmmu_val_32k \
      --output_dir outputs/analysis/error_taxonomy

Inputs: predictions.jsonl + scores_qwen.csv (Qwen parser, --extract final) of each run.
Outputs: r1_wrong_classified.csv, flips_r1_r2.csv, summary.json

Rules are applied in order; the first matching one wins. Cases that need reading the response are fixed in
MANUAL_* below with a one-line reason, so the output is reproducible.
"""
import argparse
import collections
import csv
import json
import os
import string
import zlib

csv.field_size_limit(10 ** 9)


# ---------------------------------------------------------------- loop detection
def zlib_ratio(text):
    """Compressed/raw size of the last 10k chars (BASELINE_NOTES: < 0.10 = repetition loop)."""
    t = text[-10000:].encode()
    return len(zlib.compress(t)) / len(t)


def repeated_line_ratio(text):
    """Share of lines (>= 15 chars) in the last 10k chars that occur 2+ times verbatim in the whole response."""
    lines = [ln.strip() for ln in text.split('\n') if len(ln.strip()) >= 15]
    cnt = collections.Counter(lines)
    tail = [ln.strip() for ln in text[-10000:].split('\n') if len(ln.strip()) >= 15]
    return sum(cnt[ln] >= 2 for ln in tail) / max(1, len(tail))


def status(rec):
    """finished / loop / trunc_calc. loop = zlib < 0.10 (notes' criterion) OR >= 80% of tail lines are verbatim repeats."""
    if rec['finish_reason'] != 'length':
        return 'finished'
    r = rec['response']
    return 'loop' if zlib_ratio(r) < 0.10 or repeated_line_ratio(r) >= 0.8 else 'trunc_calc'


# ---------------------------------------------------------------- manual judgments (read from the responses)
# Task 1, finished open questions (34 wrong). Default for anything not listed = 'open_wrong'.
MANUAL_OPEN = {
    # notation: extracted final answer has the gold value, the parser can't match the form
    'validation_Basic_Medical_Science_10': ('notation', "answers 'c' (correct); open gold 'C' is a letter -> MMMU_preproc maps the response to A, gt stays 'C' -> can never match"),
    'validation_Biology_10': ('notation', "\\dfrac{1}{64} vs 1/64"),
    'validation_Electronics_2': ('notation', "2\\sqrt{2} (=2.828) vs 2.83"),
    'validation_Electronics_27': ('notation', "240 deg vs -120 deg: same angle"),
    'validation_Finance_17': ('notation', "1,000 vs 1000 (thousands separator)"),
    'validation_Finance_20': ('notation', "7,243,000 vs 7243000 (thousands separator)"),
    'validation_Finance_30': ('notation', "1,249.24 vs 1249 (separator; gold is the rounded value)"),
    'validation_Manage_20': ('notation', "'2,960 unfavorable' vs 2960 (separator + trailing word)"),
    'validation_Math_15': ('notation', "\\frac{24}{7} vs ['24/7','3.429'] (list gold is never matched by the Qwen scorer either)"),
    # extraction failure: correct answer is in the response, extractor took the wrong span
    'validation_Electronics_25': ('extract_fail', "'Answer:' line is a sentence, the value '50V' is on the next line -> extractor keeps only the sentence"),
    # ambiguous
    'validation_Chemistry_13': ('ambiguous', "13.0 vs 12.97: model value is coarser than gold (3 s.f.) - rounding or a small arithmetic slip (it computed 13.006)"),
    'validation_Chemistry_25': ('ambiguous', "24.3 vs 24.32: model value is coarser than gold"),
    'validation_Finance_23': ('ambiguous', "527.74 vs 527.89 (0.03%): same method, intermediate rounding differs"),
    'validation_Manage_3': ('ambiguous', "242,159.73 vs 242,110.62 (0.02%): close, intermediate rounding or a small error"),
    'validation_Geography_4': ('ambiguous', "answers 'Clearwater, Florida' vs gold ['Tampa','Florida']: half right; list gold can't match in the Qwen scorer"),
}

# Task 1, finished multiple-choice that did not come out as a clean single letter (17 of 229).
# Default for everything else (clean letter != gold) = 'mc_wrong'.
MANUAL_MC = {
    'validation_Agriculture_12': ('no_answer', "explicitly picks no option ('no option is correct')"),
    'validation_Math_24': ('notation', "'Final Answer: r = \\frac{2R}{3}' is option A in LaTeX; 'Correct Option: A' comes later but is not a marker"),
    # extraction also failed, but the model's own answer is a wrong option -> still mc_wrong (flag extract_fail)
    'validation_Agriculture_23': ('mc_wrong', "no marker; ends with '**B. Herbicide damage**' (gold D)", 'extract_fail'),
    'validation_Chemistry_26': ('mc_wrong', "'I'll go with B' (gold C); extractor took 'set (b).'", 'extract_fail'),
    'validation_Diagnostics_and_Laboratory_Medicine_14': ('mc_wrong', "no marker; ends with '**B. Syringomyelia**' (gold A)", 'extract_fail'),
    'validation_Manage_6': ('mc_wrong', "says the question is broken, then 'if forced: B' (gold D)", 'extract_fail'),
}

# Task 1, truncated responses that are neither verbatim loops nor clearly progressing.
MANUAL_TRUNC = {
    'validation_Electronics_9': ('ambiguous', "trunc: cycles through guessed waveforms/series without verbatim repetition"),
    'validation_Math_12': ('ambiguous', "trunc: keeps re-trying constraint readings, all ending 'objective 0'"),
    'validation_Mechanical_Engineering_4': ('ambiguous', "trunc: 'Perhaps... Not matching' cycle; a few blocks repeat verbatim but below the 80% line"),
}

# Task 2, both runs finished and correctness flipped, but the scorer saw different things for the same answer.
MANUAL_FLIP_SAME_ANSWER = {
    'validation_Electronics_25': "both answer 50 V; run1 extractor took the sentence on the 'Answer:' line, run2 \\boxed{50 V}",
    'validation_Finance_20': "both 7,243,000; run1 '7,243,000' (separator, fail), run2 \\boxed{7243000}",
    'validation_Basic_Medical_Science_14': "both answer 'genetic engineering / recombinant DNA'; run1 had no marker so the full response was parsed and matched the word 'Transformation' in a step list (false positive); run2 marker -> fail",
}

CAT_KO = {
    'trunc_loop': '1. 응답 잘림 - 반복 루프',
    'trunc_calc': '1. 응답 잘림 - 루프 없이 계산 중',
    'no_answer': '2. 최종 답 없음',
    'extract_fail': '2. 추출 실패 (답은 맞음)',
    'notation': '3. 표기 차이',
    'mc_wrong': '4. 객관식 오답',
    'open_wrong': '5. 주관식 오답',
    'ambiguous': '6. 애매 (따로 모음)',
}


def load(d):
    preds = {}
    with open(os.path.join(d, 'predictions.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            preds[r['id']] = r
    with open(os.path.join(d, 'scores_qwen.csv')) as f:
        scores = {r['id']: r for r in csv.DictReader(f)}
    return preds, scores


def gold_text(p):
    if p['options']:
        return f"{p['answer']}. {p['options'][string.ascii_uppercase.index(p['answer'])]}"
    return p['answer']


def classify_wrong(p, s):
    """-> (category, note, flags)"""
    st = status(p)
    if st != 'finished':
        if p['id'] in MANUAL_TRUNC:
            return MANUAL_TRUNC[p['id']][0], MANUAL_TRUNC[p['id']][1], 'truncated'
        return ('trunc_loop' if st == 'loop' else 'trunc_calc'), '', ''
    if not p['options']:
        cat, note = MANUAL_OPEN.get(p['id'], ('open_wrong', ''))
        return cat, note, ''
    if p['id'] in MANUAL_MC:
        m = MANUAL_MC[p['id']]
        return m[0], m[1], (m[2] if len(m) > 2 else '')
    # clean case: parser output is a letter that differs from gold
    flag = 'short_answer_no_marker' if s['extract'] == 'none' else ''
    return 'mc_wrong', '', flag


def model_answer(p, s):
    """Model's own final answer as seen in scores (letter for clean MC, else the extracted span)."""
    if status(p) != 'finished':
        return '(truncated)'
    return s['extracted'][:120]


def classify_flip(i, p1, s1, p2, s2):
    """-> (direction, reason, detail)"""
    up = s2['correct'] == 'True'
    d = 'W->R' if up else 'R->W'
    a, b = status(p1), status(p2)
    n2 = int(p2['num_output_tokens'])
    if a != 'finished' and b == 'finished':
        if n2 > 16384:
            return d, '1차 잘림 -> 2차 16384 넘겨 완주 (예산 효과)', f'run2 {n2} tokens'
        return d, '1차 잘림 -> 2차 루프 없이 완주', f'run1 {a}, run2 finished at {n2} tokens (< 16384)'
    if a == 'finished' and b != 'finished':
        if up:
            return d, '2차 루프 - 루프 속 문장이 우연히 정답', f"run2 extracted '{s2['extracted'][:50]}'"
        return d, '2차에 루프에 빠짐', f"run1 finished at {p1['num_output_tokens']} tokens"
    if a != 'finished' and b != 'finished':
        which = '2차' if up else '1차'
        return d, '둘 다 루프 - 우연 일치 여부만 다름', f'{which} loop text matched by chance'
    if i in MANUAL_FLIP_SAME_ANSWER:
        return d, '둘 다 완주, 답은 같고 추출·표기가 달라짐', MANUAL_FLIP_SAME_ANSWER[i]
    return d, '둘 다 완주, 최종 답이 달라짐', f"run1 '{model_answer(p1, s1)[:40]}' -> run2 '{model_answer(p2, s2)[:40]}'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run1', required=True)
    ap.add_argument('--run2', required=True)
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()
    P1, S1 = load(args.run1)
    P2, S2 = load(args.run2)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- task 1
    rows = []
    for i, s in S1.items():
        if s['correct'] == 'True':
            continue
        p = P1[i]
        cat, note, flags = classify_wrong(p, s)
        rows.append({'id': i, 'subject': p['subject'], 'question_type': p['question_type'],
                     'category': CAT_KO[cat], 'category_key': cat, 'flags': flags, 'note': note,
                     'gold': gold_text(p)[:150], 'extract': s['extract'], 'extracted': s['extracted'][:200],
                     'parsed': s['parsed'], 'method': s['method'], 'finish_reason': p['finish_reason'],
                     'num_output_tokens': p['num_output_tokens'],
                     'zlib_last10k': round(zlib_ratio(p['response']), 4),
                     'repeated_line_ratio': round(repeated_line_ratio(p['response']), 3),
                     'run2_correct': S2[i]['correct']})
    rows.sort(key=lambda r: (r['category'], r['id']))
    with open(os.path.join(args.output_dir, 'r1_wrong_classified.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- task 2
    frows = []
    for i in S1:
        if S1[i]['correct'] == S2[i]['correct']:
            continue
        d, reason, detail = classify_flip(i, P1[i], S1[i], P2[i], S2[i])
        r1cat = next((r['category'] for r in rows if r['id'] == i), '(1차 정답)')
        frows.append({'id': i, 'subject': P1[i]['subject'], 'question_type': P1[i]['question_type'],
                      'direction': d, 'reason': reason, 'detail': detail, 'gold': gold_text(P1[i])[:100],
                      'r1_status': status(P1[i]), 'r1_tokens': P1[i]['num_output_tokens'],
                      'r1_answer': model_answer(P1[i], S1[i]), 'r2_status': status(P2[i]),
                      'r2_tokens': P2[i]['num_output_tokens'], 'r2_answer': model_answer(P2[i], S2[i]),
                      'r1_task1_category': r1cat})
    frows.sort(key=lambda r: (r['direction'], r['reason'], r['id']))
    with open(os.path.join(args.output_dir, 'flips_r1_r2.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(frows[0]))
        w.writeheader()
        w.writerows(frows)

    summary = {
        'task1_total_wrong': len(rows),
        'task1_by_category': dict(sorted(collections.Counter(r['category'] for r in rows).items())),
        'task1_by_category_and_type': {f'{k[0]} | {k[1]}': v for k, v in sorted(
            collections.Counter((r['category'], r['question_type']) for r in rows).items())},
        'task1_flags': dict(collections.Counter(r['flags'] for r in rows if r['flags'])),
        'task2_total_flips': len(frows),
        'task2_by_reason': {f'{k[0]} | {k[1]}': v for k, v in sorted(
            collections.Counter((r['direction'], r['reason']) for r in frows).items())},
        'status_counts': {'run1': dict(collections.Counter(status(p) for p in P1.values())),
                          'run2': dict(collections.Counter(status(p) for p in P2.values()))},
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
