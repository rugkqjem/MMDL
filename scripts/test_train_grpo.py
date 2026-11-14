"""CPU checks for train_grpo.py without the model download: image resizing, dataset rows, GRPO/LoRA configs.
Parity with the real processor (chat template, no second resize) runs on the server: train_grpo.py --check_only.
Run: python scripts/test_train_grpo.py   (needs the training env: trl, peft, torch, qwen-vl-utils, datasets)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from qwen_vl_utils import smart_resize

import train_grpo as tg
from rl_reward import make_reward_func

PATCH = 16  # Qwen3-VL image_processor.patch_size; qwen_vl_utils resizes to multiples of 2 * patch
tmp = tempfile.mkdtemp()
os.makedirs(os.path.join(tmp, 'images'))
sizes = {'small': (120, 90), 'big': (3000, 2400), 'wide': (2048, 300), 'mid': (900, 700)}
for name, wh in sizes.items():
    Image.new('RGBA' if name == 'mid' else 'RGB', wh, 'white').save(os.path.join(tmp, 'images', f'{name}.png'))


def row(i, names):
    prompt = f'Question: q{i}\nOptions:\nA. x\nB. y\nC. z\nPlease select the correct answer from the options above.'
    return {'id': f'r{i}', 'problem': '<image>' * len(names) + prompt, 'prompt': prompt,
            'images': [f'images/{n}.png' for n in names], 'options': ['x', 'y', 'z'], 'answer': 'B',
            'question_type': 'multiple-choice', 'category': 'test'}


rows = [row(0, ['small']), row(1, ['big', 'wide']), row(2, ['mid', 'small', 'big'])]

# 1) images are resized exactly like the eval (qwen_vl_utils.smart_resize with the eval min/max pixels)
ds = tg.build_dataset(rows, tmp, PATCH)
for r, d in zip(rows, ds):
    assert len(d['images']) == len(r['images'])
    for path, im in zip(r['images'], d['images']):
        w0, h0 = sizes[os.path.basename(path)[:-4]]
        h, w = smart_resize(h0, w0, factor=2 * PATCH, min_pixels=tg.EVAL.min_pixels, max_pixels=tg.EVAL.max_pixels)
        assert im.size == (w, h) and im.mode == 'RGB', (path, im.size, (w, h), im.mode)
        assert w % (2 * PATCH) == 0 and h % (2 * PATCH) == 0

# 2) TRL row: image placeholders first, then the eval prompt text; reward columns carried along
d = ds[1]
assert d['prompt'][0]['role'] == 'user'
assert [c['type'] for c in d['prompt'][0]['content']] == ['image', 'image', 'text']
assert d['prompt'][0]['content'][-1]['text'] == rows[1]['prompt'] and d['answer'] == 'B'
bad = dict(rows[0], problem='<image><image>' + rows[0]['prompt'])
try:
    tg.to_trl_row(bad, [None])
    raise AssertionError('image/tag count mismatch should raise')
except ValueError:
    pass

# 3) reward reads the dataset columns as TRL passes them
f = make_reward_func({151645, 151643})
assert f(completions=['The answer is (B).', 'The answer is (C).', 'The answer is (B).'],
         completion_ids=[[1, 151645], [1, 151645], [1, 2]], answer=ds['answer'], options=ds['options']) \
    == [1.0, 0.0, -0.5]

# 4) GRPO config: values that decide what is learned
args = tg.parse_args(['--output_dir', os.path.join(tmp, 'out')])
cpu = dict(bf16=False, use_cpu=True)  # this test machine has no GPU; the real run keeps bf16
cfg, sampling = tg.make_grpo_config(args, max_model_len=16384, **cpu)
assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.repetition_penalty) == (1.0, 1.0, 0, 1.0)
assert cfg.max_completion_length == 8192 and cfg.num_generations == 8 and not cfg.mask_truncated_completions
assert cfg.beta == 0.0 and cfg.loss_type == 'dapo' and cfg.use_vllm and cfg.vllm_mode == 'colocate'
assert cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps == 32
assert cfg.generation_batch_size % (cfg.num_generations * cfg.per_device_train_batch_size) == 0
ev, _ = tg.make_grpo_config(tg.parse_args(['--sampling', 'eval']), 16384, **cpu)
assert (ev.temperature, ev.top_p, ev.top_k) == (0.7, 0.8, 20)

# 5) LoRA lands on every LLM decoder linear and nowhere in the vision tower (tiny random Qwen3-VL)
import torch
from peft import get_peft_model
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
tiny = Qwen3VLConfig(
    text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, vocab_size=1000, rope_scaling={'rope_type': 'default',
                     'mrope_section': [2, 3, 3]}),
    vision_config=dict(hidden_size=32, intermediate_size=64, depth=2, num_heads=2, out_hidden_size=64,
                       deepstack_visual_indexes=[0, 1]))
torch.manual_seed(0)
model = get_peft_model(Qwen3VLForConditionalGeneration(tiny), tg.make_lora_config(args))
adapted = [n for n, _ in model.named_modules() if n.endswith('lora_A')]
assert len(adapted) == 2 * 7, adapted                                  # q,k,v,o,gate,up,down per layer
assert all('language_model' in n for n in adapted) and not any('visual' in n for n in adapted)
assert not any(p.requires_grad for n, p in model.named_parameters() if 'visual' in n)
print(f'train_grpo OK ({len(adapted)} LoRA linears on a 2-layer model; resized sizes '
      f'{[im.size for im in ds[2]["images"]]})')
