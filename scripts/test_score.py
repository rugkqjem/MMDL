"""Smoke check for prompt building and both scorers (CPU, no model). Run: python scripts/test_score.py"""
import json, os, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import build_messages
from score import extract_final_answer, score_mmmu, score_qwen

msgs, prompt, opts = build_messages({'question': 'Q <image 1>?', 'options': "['x', 'y']", 'image_1': 'IMG'}, 1, 2)
assert prompt == 'Question: Q <image 1>?\nOptions:\nA. x\nB. y\nPlease select the correct answer from the options above.'
assert msgs[0]['content'][0] == {'type': 'image', 'image': 'IMG', 'min_pixels': 1, 'max_pixels': 2}
assert build_messages({'question': 'Q', 'options': '[]', 'image_1': 'IMG'}, 1, 2)[1] == 'Question: Q'

mc = {'options': ['x', 'y', 'z'], 'answer': 'B'}
assert score_qwen({**mc, 'response': 'B'})[2] and score_mmmu({**mc, 'response': 'B'})[2]
assert score_qwen({**mc, 'response': 'The answer is (B) y.'})[2]
assert score_qwen({**mc, 'response': 'I cannot tell'})[1] == 'fail'           # judge stage removed -> wrong
assert score_mmmu({**mc, 'response': 'I cannot tell'})[1] == 'random'         # MMMU keeps random fallback
op = {'options': [], 'answer': '50'}
assert score_qwen({**op, 'response': '50 V'})[2] and score_mmmu({**op, 'response': '50 V'})[2]
assert score_mmmu({'options': [], 'answer': "['Tampa', 'Florida']", 'response': 'Tampa'})[2]
assert not score_qwen({**op, 'response': '36'})[2]

# final-answer extraction: last marker (\boxed{} or "answer is"/"Answer:") wins, option labels -> letter
opts = ['x', 'y', 'z', 'w']
ex = lambda r, o=opts: extract_final_answer(r, o)  # noqa: E731
cot = 'A (x) looks wrong, (C) z maybe.\n\n**Final Answer:**\n\n**B. y**'
assert ex(cot) == ('B', 'phrase')                                       # markdown dropped, next line taken
assert score_qwen({**mc, 'response': cot})[1] == 'fail'                 # full CoT: many letters -> no rule hit
assert score_qwen({**mc, 'response': cot}, ex(cot)[0])[2] and score_mmmu({**mc, 'response': cot}, ex(cot)[0])[2]
assert ex('so $\\boxed{9}$ and the answer is (D)') == ('D', 'phrase')  # phrase after boxed
assert ex('Answer: maybe D\nso $\\boxed{\\text{C}}$') == ('C', 'boxed')  # boxed after phrase
assert ex('Thus the correct answer is: C. z') == ('C', 'phrase')
assert ex('Correct option: **B**') == ('Correct option: **B**', 'none')  # no precedent marker -> upstream
assert ex('D. w') == ('D. w', 'none')                                   # no marker -> full response
assert ex('Answer: none of them') == ('none of them', 'phrase')         # not a label -> text to parser
assert ex('Final Answer: $\\boxed{24/7}$', []) == ('24/7', 'boxed')
assert ex('**Answer:**', []) == ('**Answer:**', 'none')                 # empty span -> upstream

with tempfile.TemporaryDirectory() as d:
    pred = os.path.join(d, 'p.jsonl')
    with open(pred, 'w') as f:
        for i, (s, r) in enumerate([('Art', 'B'), ('Art', 'C'), ('Math', 'B')]):
            f.write(json.dumps({'id': str(i), 'subject': s, 'question_type': 'multiple-choice', 'response': r,
                                'finish_reason': 'stop', **mc}) + '\n')
    for parser in ('qwen', 'mmmu'):
        for extract, name in (('final', parser), ('none', f'{parser}_raw')):
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'score.py'), '--pred', pred,
                            '--parser', parser, '--extract', extract, '--output_dir', d],
                           check=True, capture_output=True)
            assert json.load(open(os.path.join(d, f'scores_{name}.json')))['macro_acc'] == 75.0  # (50 + 100) / 2
print('ok')
