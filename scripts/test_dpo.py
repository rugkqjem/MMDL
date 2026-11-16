"""CPU checks for the DPO pipeline (loop_onset, dpo_pairs, train_dpo) without model downloads.
What needs the real processor (image grid, token lengths, EOS id) is checked on the server by
`train_dpo.py --check_only`. Run: python scripts/test_dpo.py   (needs trl, peft, datasets in the env)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpo_pairs as dp
from loop_onset import branch_candidates, dense_onset, find_onset, loop_share, unit_flags, units

# ---------------------------------------------------------------- loop_onset
intro = 'Intro line about the problem.\n' + '\n'.join(
    f'Step {i}: compute the area of triangle ABC with base {i} and height 3.' for i in range(30)) + '\n'
block = ('Wait, let me recheck the calculation of the area of the triangle carefully once more.\n'
         'The area is 1/2 * 6 * 4 = 12, which is not among the options.\n')
text = intro + block * 60
assert all(text[a:b] == s for a, b, s in units(text))
r = find_onset(text)
assert r['loop'] and r['onset_char'] == len(intro) + len(block) and r['first_copy_char'] == len(intro), r
br = branch_candidates(text)
assert br == sorted(br, reverse=True) and len(br) == 4 and all(text[b - 1] == '\n' for b in br)
assert abs(br[0] - 0.6 * len(text)) < 200 and abs(br[-1] - 0.15 * len(text)) < 200
d = dense_onset(text)
assert d is not None and len(intro) <= d <= len(intro) + 2 * len(block), d     # density rule near the loop
fl = unit_flags(text)
assert loop_share(fl, 0, len(intro)) < 0.2 and loop_share(fl, len(intro) + len(block), len(text)) > 0.9

hedge = intro + ''.join(f'Wait, perhaps the angle at vertex {c} is the one marked, which would give {i} degrees.\n'
                        f'Perhaps instead the figure is not to scale and side {c}D is longer than it looks.\n'
                        for i, c in zip(range(60), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' * 3))
assert dense_onset(hedge) is not None and dense_onset(hedge) <= len(intro) + 400   # restart phrases count

para = ''.join(f'Let me recompute the perimeter of the rectangle using the given side lengths again, now getting '
               f'{i} cm which does not match.\n' for i in range(80))         # paraphrase loop: number changes
r = find_onset(intro + para)
assert r['loop'] and len(intro) < r['onset_char'] <= len(intro) + 200, r

random.seed(0)
words = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma'.split()
fresh = '\n'.join(' '.join(random.choice(words) + str(random.randint(0, 999)) for _ in range(12)) for _ in range(300))
assert not find_onset(fresh)['loop']                                       # long but always new: not a loop

one_line = intro + 'the answer is approximately 0.333 and so on ' * 400       # loop without newlines
r = find_onset(one_line)
assert r['loop'] and len(intro) <= r['onset_char'] <= len(intro) + 400, r

# ---------------------------------------------------------------- dpo_pairs: correct pairs
q = {'id': 'q1', 'answer': 'B', 'options': ['x', 'y', 'z', 'w'], 'prompt': 'Question: ...', 'images': ['a.png']}
S = lambda txt, fin, n, k: {'response': txt, 'finish_reason': fin, 'num_output_tokens': n, '_k': k}
samples = [S('so the answer is (B).', 'stop', 1000, 0),           # correct
           S('so the answer is (C).', 'stop', 1100, 1),           # wrong, similar length -> pair
           S('so the answer is (D).', 'stop', 3000, 2),           # wrong, too long vs 1000
           S('I cannot tell.', 'stop', 1000, 3),                  # parse failure: never rejected
           S('so the answer is (B). wait wait', 'length', 8192, 4)]  # truncated: never chosen
pairs = dp.correct_pairs(q, samples, len_ratio=1.5, per_q=1)
assert len(pairs) == 1 and pairs[0]['meta']['chosen_sample'] == 0 and pairs[0]['meta']['rejected_sample'] == 1
assert dp.correct_pairs(q, samples[:1] + samples[2:], 1.5, 1) == []   # only a too-long or unparsable wrong left

# ---------------------------------------------------------------- dpo_pairs: loop pairs
orig = text
rs = {'id': 'q1', 'sample': 4, 'branch_char': len(intro),
      'continuations': [{'text': 'Closest option is (B).\nThe answer is (B).', 'finish_reason': 'stop', 'correct': True},
                        {'text': 'The answer is (C).', 'finish_reason': 'stop', 'correct': False},
                        {'text': 'loop ' * 50, 'finish_reason': 'length', 'correct': False}]}
pr = dp.loop_pair(q, rs, orig, random.Random(0), chosen_correct=False, min_rejected=300, min_share=0.5)
assert pr['prefix'] == orig[:len(intro)] and pr['chosen'].endswith('(B).') and not pr['rejected_eos']
assert orig[len(intro):].startswith(pr['rejected']) and pr['meta']['rejected_loop_share'] >= 0.5
assert block in pr['rejected']                                         # the rejected side contains the loop
early = dp.loop_pair(q, dict(rs, branch_char=200), orig, random.Random(0), False, 300, 0.5)
assert early['meta']['rejected_loop_share'] >= 0.5 and len(early['rejected']) > len(intro) - 200   # extended into loop
assert dp.loop_pair(q, dict(rs, branch_char=None), orig, random.Random(0), False, 300, 0.5) is None
only_wrong = dict(rs, continuations=rs['continuations'][1:])
assert dp.loop_pair(q, only_wrong, orig, random.Random(0), True, 300, 0.5) is None
assert dp.loop_pair(q, only_wrong, orig, random.Random(0), False, 300, 0.5)['chosen'] == 'The answer is (C).'

full = [dict(pairs[0], id='q1', answer='B', options=q['options']), dict(pr, id='q1', answer='B', options=q['options'])]
dp.check(full, {'q1'}, set(), {'correct': 1, 'loop': 2})
bad_loop = dict(full[1], meta=dict(full[1]['meta'], rejected_loop_share=0.1, rejected_to_end=False))
for bad, msg in ((bad_loop, 'no loop text'),
                 (dict(full[0], chosen=full[0]['chosen'] + '<|im_end|>'), 'stop token')):
    try:
        dp.check([bad], {'q1'}, set(), {'correct': 1, 'loop': 2})
        raise AssertionError('check() should fail: ' + msg)
    except AssertionError as e:
        assert msg in str(e), e
try:
    dp.check(full, {'q1'}, {'q1'}, {'correct': 1, 'loop': 2})
    raise AssertionError('excluded id should fail')
except AssertionError as e:
    assert 'q1' in str(e)

# ---------------------------------------------------------------- train_dpo: text rows and config
import train_dpo as td
row = td.to_text_row(full[1], 'CHAT<|im_start|>assistant\n')
assert row['prompt'] == 'CHAT<|im_start|>assistant\n' + full[1]['prefix']
assert row['chosen'].endswith('<|im_end|>') and not row['rejected'].endswith('<|im_end|>')
row = td.to_text_row(full[0], 'CHAT')
assert row['prompt'] == 'CHAT' and row['rejected'].endswith('<|im_end|>')

args = td.parse_args(['--model_path', 'x', '--output_dir', '/tmp/dpo_test'])
cfg = td.make_dpo_config(args, bf16=False, use_cpu=True)
assert cfg.loss_type == ['sigmoid', 'sft'] and cfg.loss_weights == [1.0, 1.0] and cfg.beta == 0.1
assert not cfg.precompute_ref_log_probs and cfg.max_length is None and cfg.gradient_checkpointing
cfg = td.make_dpo_config(td.parse_args(['--model_path', 'x', '--sft_weight', '0']), bf16=False, use_cpu=True)
assert cfg.loss_type == ['sigmoid']
print('dpo pipeline OK')
