"""GRPO reward = the eval's own scorer + a truncation penalty. CPU only (numpy), no model needed.

Scoring is not re-implemented: extract_final_answer() + score_qwen() from score.py, i.e. exactly
resample_wrong.is_correct(), the primary MMMU metric. test_rl_reward.py asserts the two agree.

Reward per completion (truncation is checked BEFORE scoring):
    truncated (last token is not a stop token)         -> -trunc_penalty
    finished and correct                               ->  1.0
    finished and wrong / unparsable                    ->  0.0
A truncated response that already wrote "answer is (B)" mid-loop still gets the penalty; in the pilot rollouts
810 of 1,438 truncated responses had an answer marker and 258 would have been scored correct.

Multiple-choice only. score_qwen() grades open questions by "prediction contains the gold string" ("13" matches
gold "3"), a reward hack in RL, so rows without options raise instead of being scored silently.

Truncation = TRL's own definition (trl 1.17 grpo_trainer: `completions/clipped_ratio`, mask_truncated_completions):
the last completion token is not in eos_token_ids + [pad_token_id], whatever the length. vLLM also stops short of
max_completion_length when prompt + completion reaches max_model_len (finish_reason 'length'); a length test
(len >= max_completion_length) missed those and scored them as finished (pilot 2026-11-13: 11 of 640).

Use with TRL (the dataset needs `answer` and `options` columns; TRL passes every column except `prompt`):
    from rl_reward import make_reward_func, trl_stop_ids
    reward = make_reward_func(trl_stop_ids(processor.tokenizer, model.generation_config), trunc_penalty=0.5)
    GRPOTrainer(..., reward_funcs=reward, args=GRPOConfig(mask_truncated_completions=False, ...))
    # False: masking would drop the loop signal
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import extract_final_answer, score_qwen  # noqa: E402

DEFAULT_TRUNC_PENALTY = 0.5


def trl_stop_ids(tokenizer, generation_config):
    """The ids TRL counts as a proper end: generation_config.eos_token_id (+ tokenizer EOS) and the pad token.
    Mirrors GRPOTrainer.eos_token_ids and the `eos_and_pad` list of its clipped_ratio."""
    eos = generation_config.eos_token_id
    eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    ids = set(eos) | {tokenizer.eos_token_id, tokenizer.pad_token_id}
    ids.discard(None)
    return ids


def is_truncated(completion_ids, stop_ids):
    """TRL's definition: the completion did not end on a stop token (vLLM keeps the stop token in its output).
    Covers both cut-offs: max_completion_length and max_model_len (prompt + completion)."""
    return not (len(completion_ids) and int(completion_ids[-1]) in stop_ids)


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


def make_reward_func(stop_ids, trunc_penalty=DEFAULT_TRUNC_PENALTY):
    """TRL GRPOTrainer reward function. Logs accuracy / truncation / parse-failure rates per generation batch."""
    stop_ids = set([stop_ids] if isinstance(stop_ids, int) else stop_ids)
    if not stop_ids:
        raise ValueError('stop_ids is empty')

    def mmmu_reward(completions, completion_ids, answer, options, log_metric=None, log_extra=None, **kwargs):
        out = [score_response(_text(c), a, o, is_truncated(ids, stop_ids), trunc_penalty)
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
