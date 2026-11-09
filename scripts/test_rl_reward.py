"""Smoke check for rl_reward.py (CPU, no model). Run: python scripts/test_rl_reward.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resample_wrong import is_correct
from rl_reward import is_truncated, make_reward_func, score_response

EOS = {151645, 151643}
MAX = 8
opts = ['x', 'y', 'z', 'w']

# truncation: only "hit the limit without EOS" counts
assert not is_truncated([1, 2, 3], MAX, EOS)                       # stopped early (EOS kept or stripped)
assert not is_truncated([1] * 7 + [151645], MAX, EOS)              # EOS exactly at the limit
assert is_truncated([1] * 8, MAX, EOS)                             # cut off
assert is_truncated([1] * 9, MAX, EOS)

# reward levels; truncation decided before scoring
ok, bad = 'Step by step...\nThe answer is (B).', 'Step by step...\nThe answer is (C).'
assert score_response(ok, 'B', opts, False)['reward'] == 1.0
assert score_response(bad, 'B', opts, False)['reward'] == 0.0
assert score_response('no idea', 'B', opts, False)['reward'] == 0.0
r = score_response(ok, 'B', opts, True, trunc_penalty=0.5)        # right answer written, then looped
assert r['reward'] == -0.5 and r['correct_eval'] and not r['correct']
assert score_response(ok, 'B', opts, True, trunc_penalty=0.0)['reward'] == 0.0

# same verdict as the eval (resample_wrong.is_correct = extract_final_answer + score_qwen)
cases = [ok, bad, 'no idea', 'A (x) looks wrong, (C) z maybe.\n\n**Final Answer:**\n\n**B. y**',
         'so $\\boxed{9}$ and the answer is (D)', '\\boxed{120^\\circ}\nCorrect option: **B. 120°**',
         '✅ 正确答案：**B. y**']
for c in cases:
    assert score_response(c, 'B', opts, False)['correct_eval'] == bool(is_correct({'answer': 'B', 'options': opts}, c)), c

# open questions are refused
try:
    score_response('3', '3', [], False)
    raise AssertionError('open question should raise')
except ValueError:
    pass

# TRL-style call: standard and conversational completions, extra kwargs ignored, metrics logged
logged = {}
f = make_reward_func(MAX, EOS, trunc_penalty=0.5)
rewards = f(completions=[ok, [{'role': 'assistant', 'content': bad}], ok],
            completion_ids=[[1, 151645], [1, 2, 151645], [1] * MAX],
            answer=['B', 'B', 'B'], options=[opts, opts, opts], prompts=['p'] * 3, trainer_state=None,
            question_type=['multiple-choice'] * 3, log_metric=lambda k, v: logged.__setitem__(k, v))
assert rewards == [1.0, 0.0, -0.5], rewards
assert abs(logged['reward_parts/truncated'] - 1 / 3) < 1e-9 and abs(logged['reward_parts/correct_eval'] - 2 / 3) < 1e-9
assert f.__name__ == 'mmmu_reward'
print('rl_reward OK')
