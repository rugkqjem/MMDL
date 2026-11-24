"""Merge a GRPO LoRA adapter (train_grpo.py checkpoint) into the pinned base model -> full bf16 checkpoint for infer.py.

  python scripts/merge_lora.py --adapter outputs/grpo/run1/checkpoint-200 --out outputs/grpo/run1/merged_200
  bash scripts/run_mmmu_eval.sh --model_path outputs/grpo/run1/merged_200 --model_revision '' \\
      --data_root $HF_DATA --output_dir outputs/grpo_run1_200_mmmu_val
The processor and generation_config are copied from the base, so the eval prompt and stop tokens do not change.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--adapter', required=True, help='dir with adapter_config.json')
    p.add_argument('--out', required=True)
    p.add_argument('--model_path', default='Qwen/Qwen3-VL-4B-Instruct')
    p.add_argument('--model_revision', default=infer.MODEL_REVISION)
    a = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration
    revision = None if os.path.isdir(a.model_path) else (a.model_revision or None)
    base = Qwen3VLForConditionalGeneration.from_pretrained(a.model_path, revision=revision, dtype=torch.bfloat16)
    n_base = sum(p.numel() for p in base.parameters())
    model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    assert sum(p.numel() for p in model.parameters()) == n_base, 'merge left extra parameters'
    model.config.use_cache = True
    model.save_pretrained(a.out)
    AutoProcessor.from_pretrained(a.model_path, revision=revision).save_pretrained(a.out)
    GenerationConfig.from_pretrained(a.model_path, revision=revision).save_pretrained(a.out)
    with open(os.path.join(a.out, 'merge_info.json'), 'w') as f:
        json.dump({'adapter': os.path.abspath(a.adapter), 'base': a.model_path, 'base_revision': revision}, f, indent=2)
    print(f'merged {a.adapter} -> {a.out}')


if __name__ == '__main__':
    main()
