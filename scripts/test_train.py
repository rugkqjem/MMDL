"""CPU smoke test of the fine-tuning pipeline (~1 min): does every ablation arm train, touch only its weights, and save
a checkpoint that reloads? Uses a tiny random-weight Qwen3-VL built from the pinned config (downloads only the
config / tokenizer / processor JSONs, no weights) and three toy samples. Needs torch, transformers, peft, pillow.

    python scripts/test_train.py [--workdir /tmp/mmdl_train_test]

It does not cover vLLM evaluation (GPU only); run_mmmu_eval.sh is unchanged and already tested by test_score.py.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODEL_ID, MODEL_REVISION = 'Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
ARMS = {'deepstack': {'deepstack_merger_list'}, 'lora': {'language_model'},
        'both': {'deepstack_merger_list', 'language_model'}}


def build_tiny_model(path):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    meta = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=['*.json', '*.txt', '*.jinja'])
    cfg = AutoConfig.from_pretrained(meta)
    v, t = cfg.vision_config, cfg.text_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 6, 64, 128, 4, 64
    v.deepstack_visual_indexes = [1, 3, 5]
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 4
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    (getattr(t, 'rope_parameters', None) or t.rope_scaling)['mrope_section'] = [4, 2, 2]  # sums to head_dim / 2
    torch.manual_seed(0)
    Qwen3VLForConditionalGeneration(cfg).to(torch.bfloat16).save_pretrained(path)
    AutoProcessor.from_pretrained(meta).save_pretrained(path)


def build_toy_data(root):
    os.makedirs(f'{root}/img', exist_ok=True)
    for i, c in enumerate(['red', 'blue', 'green']):
        Image.new('RGB', (64 + 32 * i, 48), c).save(f'{root}/img/{i}.png')
    rows = [  # one placeholder image / two images without placeholders + multi-turn / system + inline placeholders
        {'image': '0.png', 'conversations': [{'from': 'human', 'value': '<image>\nWhat color?'},
                                             {'from': 'gpt', 'value': 'Red.'}]},
        {'image': ['1.png', '2.png'], 'conversations': [
            {'from': 'human', 'value': 'Compare these.'}, {'from': 'gpt', 'value': 'Blue and green.'},
            {'from': 'human', 'value': 'Which is left?'}, {'from': 'gpt', 'value': 'Blue.'}]},
        {'image': ['2.png', '0.png'], 'conversations': [
            {'from': 'system', 'value': 'Be brief.'}, {'from': 'human', 'value': 'First <image> then <image> ok?'},
            {'from': 'gpt', 'value': 'Green, red.'}]},
    ]
    with open(f'{root}/train.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return rows


def test_labels(model_dir, data_root, rows):
    from transformers import AutoProcessor
    import train_ft as T
    proc = AutoProcessor.from_pretrained(model_dir)
    ds = T.SFTDataset(rows, proc, f'{data_root}/img', 8192)
    supervised = [proc.tokenizer.decode(e['input_ids'][e['labels'] != T.IGNORE_INDEX]) for e in (ds[i] for i in range(3))]
    assert supervised == ['Red.<|im_end|>', 'Blue and green.<|im_end|>Blue.<|im_end|>', 'Green, red.<|im_end|>'], supervised
    batch = T.make_collator(proc.tokenizer.pad_token_id)([ds[i] for i in range(3)])
    assert batch['input_ids'].shape[0] == 3 and batch['image_grid_thw'].shape[0] == 5
    print('labels / collator: ok')


def test_prepare_row(workdir):
    from prepare_mmmu_sft import to_row
    sample = {'id': 'validation_Math_1', 'question': 'Area of <image 1>?', 'options': "['1', '2']", 'answer': 'B',
              'image_1': Image.new('RGB', (8, 8)), **{f'image_{i}': None for i in range(2, 8)}}
    os.makedirs(f'{workdir}/prep', exist_ok=True)
    row = to_row(sample, f'{workdir}/prep')
    assert row['image'] == ['validation_Math_1_1.png'] and row['conversations'][1]['value'] == 'Answer: B'
    assert row['conversations'][0]['value'].startswith('Question: Area of') and 'B. 2' in row['conversations'][0]['value']
    print('prepare_mmmu_sft row: ok')


def test_ablation(model_dir, data_root, out_root):
    from safetensors.torch import load_file
    from transformers import Qwen3VLForConditionalGeneration
    subprocess.run(['bash', f'{HERE}/run_ablation.sh', '--model_path', model_dir, '--train_data', f'{data_root}/train.jsonl',
                    '--image_root', f'{data_root}/img', '--out_root', out_root, '--skip_eval',
                    '--max_steps', '1', '--warmup_steps', '0', '--gradient_accumulation_steps', '1',
                    '--per_device_train_batch_size', '3', '--learning_rate', '1e-2', '--lora_learning_rate', '1e-2',
                    '--min_pixels', '1024', '--max_pixels', '4096', '--dataloader_num_workers', '0',
                    '--gradient_checkpointing', '--use_cpu'], check=True)
    base = load_file(f'{model_dir}/model.safetensors')
    for arm, expected in ARMS.items():
        ckpt = f'{out_root}/ckpt_{arm}'
        new = load_file(f'{ckpt}/model.safetensors')
        assert new.keys() == base.keys(), f'{arm}: key mismatch {set(new) ^ set(base)}'  # LoRA merged away
        assert all(t.dtype == torch.bfloat16 for t in new.values()), f'{arm}: not all bf16'
        changed = [k for k in base if not torch.equal(base[k], new[k])]
        touched = {g for g in ('deepstack_merger_list', 'language_model') if any(g in k for k in changed)}
        stray = [k for k in changed if not any(g in k for g in expected)]
        assert touched == expected and not stray, f'{arm}: touched {touched}, stray {stray[:3]}'
        if 'language_model' in expected:  # LoRA targets only decoder-layer projections
            assert all('_proj.weight' in k for k in changed if 'language_model' in k), arm
        Qwen3VLForConditionalGeneration.from_pretrained(ckpt)
        print(f'arm {arm}: ok ({len(changed)} tensors changed, reloads)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workdir', default='/tmp/mmdl_train_test')
    args = p.parse_args()
    shutil.rmtree(args.workdir, ignore_errors=True)
    model_dir, data_root = f'{args.workdir}/tiny_model', f'{args.workdir}/data'
    build_tiny_model(model_dir)
    rows = build_toy_data(data_root)
    test_labels(model_dir, data_root, rows)
    test_prepare_row(args.workdir)
    test_ablation(model_dir, data_root, f'{args.workdir}/ablation')
    print('all ok')


if __name__ == '__main__':
    main()
