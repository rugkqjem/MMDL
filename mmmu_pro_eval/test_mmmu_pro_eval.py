"""CPU smoke test (no model, no dataset download): prompt/image layout, 10-option scoring, compare.py.
Run: python mmmu_pro_eval/test_mmmu_pro_eval.py"""
import csv
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))
from infer_standard10 import build_messages  # noqa: E402
from score import extract_final_answer, score_qwen  # noqa: E402
from compare import sign_test  # noqa: E402

opts10 = [f'o{i}' for i in range(10)]
s = {'question': 'Look at <image 1> and <image 2>. Which is larger?', 'options': str(opts10[:9] + ['<image 3>']),
     'image_1': 'I1', 'image_2': 'I2', 'image_3': 'I3', 'image_4': 'I4'}
msgs, prompt, options, order = build_messages(s, 'interleave', 1, 2)
c = msgs[0]['content']
kinds = [x['type'] if x['type'] == 'text' else x['image'] for x in c]
assert kinds == ['I4', 'text', 'I1', 'text', 'I2', 'text', 'I3', 'text'], kinds  # unreferenced I4 goes first
assert c[1]['text'] == 'Look at ' and c[3]['text'] == ' and '
assert c[5]['text'].endswith('I. o8\nJ. ') and c[7]['text'].startswith('\nAnswer the preceding')
assert order == [4, 1, 2, 3] and len(options) == 10 and prompt.count('\n') == 11
assert ''.join(x.get('text', '') for x in c) == prompt.replace('<image 1>', '').replace('<image 2>', '').replace('<image 3>', '')

# repeated placeholder: inserted once, second mention stays as text; missing image stays as text
s2 = {'question': '<image 1> vs <image 1> vs <image 5>', 'options': "['a', 'b']", 'image_1': 'I1'}
c2 = build_messages(s2, 'interleave', 1, 2)[0][0]['content']
assert [x.get('image') for x in c2] == ['I1', None] and c2[1]['text'].startswith(' vs <image 1> vs <image 5>')

c3, _, _, order3 = build_messages(s, 'front', 1, 2)
assert [x.get('image') for x in c3[0]['content']] == ['I1', 'I2', 'I3', 'I4', None] and order3 == [1, 2, 3, 4]

# 10-option scoring with the repo scorer (letters up to J)
rec = {'options': opts10, 'answer': 'J'}
resp = 'Option A is wrong, maybe C...\nAnswer: J'
assert extract_final_answer(resp, opts10) == ('J', 'phrase')
assert score_qwen({**rec, 'response': resp}, 'J')[2]
assert not score_qwen({**rec, 'response': 'Answer: I'}, extract_final_answer('Answer: I', opts10)[0])[2]

assert abs(sign_test(0, 0) - 1) < 1e-12 and abs(sign_test(5, 10) - 1) < 1e-12 and sign_test(0, 10) < 0.002

# compare.py end-to-end on fake score files
with tempfile.TemporaryDirectory() as d:
    def write(name, rows):
        os.makedirs(f'{d}/{name}')
        with open(f'{d}/{name}/scores_qwen.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['id', 'subject', 'answer', 'parsed', 'method', 'correct', 'finish_reason'])
            w.writeheader()
            w.writerows(rows)
    row = lambda i, subj, ok, fin: {'id': i, 'subject': subj, 'answer': 'A', 'parsed': 'A' if ok else 'B',  # noqa: E731
                                    'method': 'rule', 'correct': str(ok), 'finish_reason': fin}
    write('a', [row('1', 'Math', True, 'stop'), row('2', 'Math', False, 'length'), row('3', 'Art', False, 'stop')])
    write('b', [row('1', 'Math', True, 'stop'), row('2', 'Math', True, 'stop'), row('3', 'Art', True, 'stop')])
    subprocess.run([sys.executable, os.path.join(HERE, 'compare.py'), '--a', f'{d}/a', '--b', f'{d}/b',
                    '--out', f'{d}/cmp'], check=True, capture_output=True)
    out = json.load(open(f'{d}/cmp/compare.json'))
    assert out['baseline']['correct'] == 1 and out['v1b']['correct'] == 3
    assert out['paired']['b_gains'] == 2 and out['paired']['b_losses'] == 0 and out['paired']['trunc_fixed'] == 1
    assert abs(out['baseline']['macro'] - 25.0) < 1e-9  # Math 50, Art 0

print('ok')
