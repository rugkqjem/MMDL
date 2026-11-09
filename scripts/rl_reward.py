"""GRPO reward = the eval's own scorer + a truncation penalty. CPU only (numpy), no model needed.

Scoring is not re-implemented: extract_final_answer() + score_qwen() from score.py, i.e. exactly
resample_wrong.is_correct(), the primary MMMU metric. test_rl_reward.py asserts the two agree.

Reward per completion (truncation is checked BEFORE scoring):
    truncated (hit max_completion_length without EOS)  -> -trunc_penalty
    finished and correct                               ->  1.0
    finished and wrong / unparsable                    ->  0.0
A truncated response that already wrote "answer is (B)" mid-loop still gets the penalty; in the pilot rollouts
810 of 1,438 truncated responses had an answer marker and 258 would have been scored correct.

Multiple-choice only. score_qwen() grades open questions by "prediction contains the gold string" ("13" matches
gold "3"), a reward hack in RL, so rows without options raise instead of being scored silently.

Use with TRL (the dataset needs `answer` and `options` columns; TRL passes every column except `prompt`):
    from rl_reward import make_reward_func
    reward = make_reward_func(max_completion_length=8192, eos_ids=eos_ids, trunc_penalty=0.5)
    GRPOTrainer(..., reward_funcs=reward, args=GRPOConfig(max_completion_length=8192,
                mask_truncated_completions=False, ...))   # False: masking would drop the loop signal
eos_ids: every id generation stops on (model.generation_config.eos_token_id, a list for Qwen3-VL).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import extract_final_answer, score_qwen  # noqa: E402

DEFAULT_TRUNC_PENALTY = 0.5


def is_truncated(completion_ids, max_completion_length, eos_ids):
    """Cut off by the length limit: reached max_completion_length and the last token is not an EOS.
    Works whether or not the trainer keeps the EOS token in completion_ids (a shorter completion always stopped)."""
    if len(completion_ids) < max_completion_length:
        return False
    return not (completion_ids and completion_ids[-1] in eos_ids)


def score_response(response, answer, options, truncated, trunc_penalty=DEFAULT_TRUNC_PENALTY):
    """One completion -> dict(reward, correct, correct_eval, truncated, extract, method, parsed).
    correct_eval: the eval's verdict on the text (ignores truncation), same as rl_rollout_filter.py.
    correct: what the reward counts as right (correct_eval and not truncated), same as rl_rollout_filter.py."""
    options = list(options or [])
    if not options:
        raise ValueError('open question: score_qwen accepts any prediction containing the gold string, '
                         'which RL would exploit; keep the RL set multiple-choice only')
    rec = {'answer': answer, 'options': options, 'response': response}
    text, how = extract_final_answer(response, options)        # = resample_wrong.is_correct()
    parsed, method, ok = score_qwen(rec, text)
    ok = bool(ok)
    if truncated:
        reward = -float(trunc_penalty)
    else:
        reward = 1.0 if ok else 0.0
    return {'reward': reward, 'correct': ok and not truncated, 'correct_eval': ok, 'truncated': bool(truncated),
            'extract': how, 'method': method, 'parsed': parsed}


def _text(completion):
    """TRL passes str (standard format) or [{'role': 'assistant', 'content': ...}] (conversational)."""
    if isinstance(completion, str):
        return completion
    content = completion[-1]['content']
    if isinstance(content, list):  # [{'type': 'text', 'text': ...}]
        return ''.join(c.get('text', '') for c in content if isinstance(c, dict))
    return content


def make_reward_func(max_completion_length, eos_ids, trunc_penalty=DEFAULT_TRUNC_PENALTY):
    """TRL GRPOTrainer reward function. Logs accuracy / truncation / parse-failure rates per generation batch."""
    eos_ids = set([eos_ids] if isinstance(eos_ids, int) else eos_ids)
    if not eos_ids:
        raise ValueError('eos_ids is empty')

    def mmmu_reward(completions, completion_ids, answer, options, log_metric=None, log_extra=None, **kwargs):
        out = [score_response(_text(c), a, o, is_truncated(ids, max_completion_length, eos_ids), trunc_penalty)
               for c, ids, a, o in zip(completions, completion_ids, answer, options)]
        n = max(len(out), 1)
        if log_metric:
            log_metric('reward_parts/correct', sum(r['correct'] for r in out) / n)
            log_metric('reward_parts/correct_eval', sum(r['correct_eval'] for r in out) / n)  # if truncation ignored
            log_metric('reward_parts/truncated', sum(r['truncated'] for r in out) / n)
            log_metric('reward_parts/parse_fail', sum((not r['truncated']) and r['method'] != 'rule' for r in out) / n)
        if log_extra:
            log_extra('gold', list(answer))
            log_extra('parsed', [r['parsed'] or '-' for r in out])
            log_extra('truncated', [r['truncated'] for r in out])
        return [r['reward'] for r in out]

    mmmu_reward.__name__ = 'mmmu_reward'  # TRL logs rewards/<name>/mean
    return mmmu_reward
