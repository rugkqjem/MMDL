"""CPU check for the vLLM prompt fix in train_grpo.py (no vLLM, no model). Run: python scripts/test_vllm_prompt_fix.py

The fake engine expands placeholders the way vLLM does for token prompts: the k-th <|image_pad|> in the prompt
becomes image k's N_k pad tokens. Fed TRL's already-expanded ids it reproduces the pilot bug (prompt grows, images
out of order); fed one pad per image (the fix) it rebuilds exactly the trainer's prompt.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_grpo import collapse_image_pads, install_vllm_prompt_fix, make_vllm_prompt_fix

PAD, VS, VE, EOS = 151655, 151652, 151653, 151645  # Qwen3-VL <|image_pad|>, <|vision_start|>, <|vision_end|>


def trainer_prompt(n_tokens, text=(11, 12, 13)):
    """What TRL hands to vLLM: every image already expanded to N pads by the processor."""
    ids = [1, 2]
    for n in n_tokens:
        ids += [VS] + [PAD] * n + [VE]
    return ids + list(text)


def fake_vllm(n_tokens_of):
    """generate(prompts) -> outputs; the k-th pad of a prompt is replaced by image k's tokens (vLLM prompt update)."""
    calls = []

    def generate(prompts, sampling_params=None, use_tqdm=False):
        calls.append(prompts)
        outs = []
        for p in prompts:
            ids, imgs = p['prompt_token_ids'], p.get('multi_modal_data', {}).get('image')
            if imgs is not None:
                sizes = n_tokens_of(imgs if isinstance(imgs, list) else [imgs])
                ids, k, out = list(ids), 0, []
                for t in ids:
                    if t == PAD and k < len(sizes):
                        out += [PAD] * sizes[k]
                        k += 1
                    else:
                        out.append(t)
                ids = out
            outs.append(SimpleNamespace(prompt_token_ids=ids, outputs=[SimpleNamespace(token_ids=[7, EOS])]))
        return outs

    return generate, calls


sizes = lambda imgs: [im['n'] for im in imgs]  # noqa: E731  fake image object carries its token count
imgs5 = [{'n': n} for n in (6, 4, 4, 5, 4)]
ids5 = trainer_prompt([6, 4, 4, 5, 4])

# 1) collapse: one pad per image, everything else untouched
short = collapse_image_pads(ids5, PAD)
assert short == [1, 2] + [VS, PAD, VE] * 5 + [11, 12, 13] and short.count(PAD) == 5
assert collapse_image_pads([1, 2, 3], PAD) == [1, 2, 3]

# 2) the pilot bug: already-expanded ids go straight to vLLM -> longer prompt (the 2026-11-13 cut-offs)
gen, _ = fake_vllm(sizes)
bad = gen([{'prompt_token_ids': ids5, 'multi_modal_data': {'image': imgs5}}])[0].prompt_token_ids
assert len(bad) > len(ids5), (len(bad), len(ids5))

# 3) with the fix: vLLM receives one pad per image and runs exactly the trainer's prompt
gen, calls = fake_vllm(sizes)
fixed = make_vllm_prompt_fix(gen, PAD)
text_only = {'prompt_token_ids': [1, 2, 3]}
one_img = {'prompt_token_ids': trainer_prompt([3]), 'multi_modal_data': {'image': {'n': 3}}}
outs = fixed([{'prompt_token_ids': ids5, 'multi_modal_data': {'image': imgs5}}, text_only, one_img],
             sampling_params=None, use_tqdm=False)
assert [o.prompt_token_ids for o in outs] == [ids5, [1, 2, 3], trainer_prompt([3])]
sent = calls[0]
assert sent[0]['prompt_token_ids'] == short and sent[0]['multi_modal_data']['image'] is imgs5
assert sent[1] is text_only and sent[2]['prompt_token_ids'].count(PAD) == 1

# 4) vLLM expanding to a different size than the trainer (e.g. its own resize) is an error, not a silent mismatch
gen, _ = fake_vllm(lambda imgs: [im['n'] + 1 for im in imgs])
try:
    make_vllm_prompt_fix(gen, PAD)([{'prompt_token_ids': ids5, 'multi_modal_data': {'image': imgs5}}])
    raise AssertionError('size mismatch should raise')
except RuntimeError as e:
    assert 'image tokens expanded differently' in str(e)

# 5) pad runs that do not map onto images are refused before reaching vLLM
try:
    make_vllm_prompt_fix(gen, PAD)([{'prompt_token_ids': ids5, 'multi_modal_data': {'image': imgs5[:4]}}])
    raise AssertionError('run/image count mismatch should raise')
except ValueError:
    pass

# 6) an engine that reports the unexpanded prompt cannot be verified: warn once, keep going
logs = []
echo = lambda prompts, **kw: [SimpleNamespace(prompt_token_ids=p['prompt_token_ids']) for p in prompts]  # noqa: E731
f = make_vllm_prompt_fix(echo, PAD, log=logs.append)
for _ in range(2):
    f([{'prompt_token_ids': ids5, 'multi_modal_data': {'image': imgs5}}])
assert len(logs) == 1 and 'not verified' in logs[0]

# 7) installed on the trainer's colocated engine
gen, calls = fake_vllm(sizes)
trainer = SimpleNamespace(vllm_generation=SimpleNamespace(llm=SimpleNamespace(generate=gen)))
install_vllm_prompt_fix(trainer, PAD)
assert trainer.vllm_generation.llm.generate.vllm_prompt_fix
trainer.vllm_generation.llm.generate([{'prompt_token_ids': ids5, 'multi_modal_data': {'image': imgs5}}])
assert calls[0][0]['prompt_token_ids'] == short
try:
    install_vllm_prompt_fix(SimpleNamespace(), PAD)
    raise AssertionError('missing engine should raise')
except RuntimeError:
    pass
print(f'vllm prompt fix OK (5-image prompt: {len(ids5)} tokens; unfixed vLLM {len(bad)}; '
      f'sent {len(short)} -> ran {len(ids5)})')
