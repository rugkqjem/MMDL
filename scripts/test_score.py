"""Smoke check for prompt building and both scorers (CPU, no model). Run: python scripts/test_score.py"""
import json, os, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import build_messages
from score import score_mmmu, score_qwen

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

with tempfile.TemporaryDirectory() as d:
    pred = os.path.join(d, 'p.jsonl')
    with open(pred, 'w') as f:
        for i, (s, r) in enumerate([('Art', 'B'), ('Art', 'C'), ('Math', 'B')]):
            f.write(json.dumps({'id': str(i), 'subject': s, 'question_type': 'multiple-choice', 'response': r,
                                'finish_reason': 'stop', **mc}) + '\n')
    for parser in ('qwen', 'mmmu'):
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'score.py'),
                        '--pred', pred, '--parser', parser, '--output_dir', d], check=True, capture_output=True)
        assert json.load(open(os.path.join(d, f'scores_{parser}.json')))['macro_acc'] == 75.0  # (50 + 100) / 2
print('ok')
