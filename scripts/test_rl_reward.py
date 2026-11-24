"""Smoke check for rl_reward.py (CPU, no model). Run: python scripts/test_rl_reward.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resample_wrong import is_correct
from types import SimpleNamespace

from rl_reward import is_truncated, make_reward_func, score_response, trl_stop_ids

EOS = {151645, 151643}
MAX = 8
opts = ['x', 'y', 'z', 'w']

# truncation = TRL clipped_ratio: last token not in eos + pad, whatever the length
assert not is_truncated([1, 2, 151645], EOS)                       # stopped early on EOS
assert not is_truncated([1] * 7 + [151645], EOS)                   # EOS exactly at the limit
assert not is_truncated([1, 151643], EOS)                          # <|endoftext|> (eos and pad)
assert is_truncated([1] * MAX, EOS)                                # cut off at max_completion_length
assert is_truncated([1, 2, 3], EOS)                                # cut off early by vLLM max_model_len (pilot)
assert is_truncated([], EOS)
import numpy as np
assert not is_truncated(np.array([1, 151645]), EOS) and is_truncated(np.array([1, 2]), EOS)

# stop ids = TRL's eos_token_ids + pad (Qwen3-VL: generation_config [151645, 151643], pad 151643)
tok = SimpleNamespace(eos_token_id=151645, pad_token_id=151643)
assert trl_stop_ids(tok, SimpleNamespace(eos_token_id=[151645, 151643])) == EOS
assert trl_stop_ids(SimpleNamespace(eos_token_id=2, pad_token_id=None), SimpleNamespace(eos_token_id=7)) == {2, 7}

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
f = make_reward_func(EOS, trunc_penalty=0.5)
rewards = f(completions=[ok, [{'role': 'assistant', 'content': bad}], ok, ok],
            completion_ids=[[1, 151645], [1, 2, 151645], [1] * MAX, [1, 2, 3]],
            answer=['B'] * 4, options=[opts] * 4, prompts=['p'] * 4, trainer_state=None,
            question_type=['multiple-choice'] * 4, log_metric=lambda k, v: logged.__setitem__(k, v))
assert rewards == [1.0, 0.0, -0.5, -0.5], rewards                  # 4th: cut short, answer written -> still penalty
assert abs(logged['reward_parts/truncated'] - 2 / 4) < 1e-9 and abs(logged['reward_parts/correct_eval'] - 3 / 4) < 1e-9
assert f.__name__ == 'mmmu_reward'
print('rl_reward OK')
