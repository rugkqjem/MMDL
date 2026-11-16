"""DPO for Qwen3-VL-4B with TRL DPOTrainer + LoRA on pairs from dpo_pairs.py (correct pairs and loop pairs).

Start from the model the pairs were sampled with (RFT/SFT merged checkpoint, or the base model): with LoRA, TRL's
reference model is that same model with the adapter switched off, so "don't move far from the start" means
"don't move far from the RFT model".

Inputs are built exactly like the eval (and train_grpo.py): eval chat template (images first, then the question),
images resized up front with the eval's qwen_vl_utils call, processor set so it does not resize again.
Text format per pair (TRL standard, non-conversational, so we decide the stop tokens):
    prompt   = chat template (... <|im_start|>assistant\\n) + prefix     prefix = shared part of a loop pair, '' otherwise
    chosen   = chosen continuation + <|im_end|>                         ends like a finished answer
    rejected = rejected continuation (+ <|im_end|> only for correct pairs; a loop never stopped)
TRL 1.17 does not append EOS for vision datasets (checked in dpo_trainer.py), so the strings above are what is trained.
Loss: sigmoid DPO + weighted SFT on chosen (loss_type ["sigmoid", "sft"], RPO), which keeps the chosen likelihood
from dropping together with the rejected one.

  python scripts/train_dpo.py --model_path outputs/rft/merged --check_only     # CPU: data, parity, lengths
  python scripts/train_dpo.py --model_path outputs/rft/merged --output_dir outputs/dpo/run1
  python scripts/merge_lora.py --model_path outputs/rft/merged --adapter outputs/dpo/run1/final_adapter \\
      --out outputs/dpo/run1/merged
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_grpo import EVAL, eval_messages, resize_like_eval  # noqa: E402

IM_END = '<|im_end|>'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pairs', default='outputs/dpo/pairs.jsonl')
    p.add_argument('--val_pairs', default=None)
    p.add_argument('--kinds', nargs='+', default=['correct', 'loop'])
    p.add_argument('--image_root', default='data/virl39k_raw')
    p.add_argument('--model_path', required=True, help='RFT merged dir (planned) or Qwen/Qwen3-VL-4B-Instruct')
    p.add_argument('--model_revision', default=None, help='default: infer.MODEL_REVISION for the hub base model')
    p.add_argument('--output_dir', default='outputs/dpo/run1')
    p.add_argument('--check_only', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    # loss
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--sft_weight', type=float, default=1.0, help='weight of the SFT term on chosen (0 = plain DPO)')
    p.add_argument('--ld_alpha', type=float, default=None, help='LD-DPO length desensitization (None = off)')
    # optimization
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=5e-6)
    p.add_argument('--num_train_epochs', type=float, default=1)
    p.add_argument('--max_steps', type=int, default=-1)
    p.add_argument('--per_device_train_batch_size', type=int, default=1)
    p.add_argument('--gradient_accumulation_steps', type=int, default=16)
    p.add_argument('--warmup_ratio', type=float, default=0.05)
    p.add_argument('--max_length', type=int, default=None, help='None = no truncation (lengths are reported first)')
    p.add_argument('--save_steps', type=int, default=25)
    p.add_argument('--report_to', default='none')
    p.add_argument('--seed', type=int, default=EVAL.seed)
    return p.parse_args(argv)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def chat_prompt(processor, prompt, n_images):
    """Eval chat template text for a question with n images (placeholders expanded later by the processor)."""
    return processor.apply_chat_template(eval_messages(['x'] * n_images, prompt, 1, 2),
                                         tokenize=False, add_generation_prompt=True)


def to_text_row(pair, chat_text):
    """One dpo_pairs.py row -> TRL standard-format strings."""
    return {'prompt': chat_text + pair['prefix'],
            'chosen': pair['chosen'] + IM_END,
            'rejected': pair['rejected'] + (IM_END if pair['rejected_eos'] else '')}


def build_dataset(pairs, processor, image_root):
    from datasets import Dataset
    ps = processor.image_processor.patch_size
    out = []
    for pr in pairs:
        imgs = resize_like_eval(pr, image_root, ps)
        out.append(to_text_row(pr, chat_prompt(processor, pr['prompt'], len(imgs))) |
                   {'images': imgs, 'id': pr['id'], 'kind': pr['kind']})
    return Dataset.from_list(out)


def check(processor, ds, eos_ids, max_length):
    """No second resize, stop token is a real EOS, token lengths (prompt incl. image tokens + completion)."""
    tok = processor.tokenizer
    im_end = tok.convert_tokens_to_ids(IM_END)
    assert im_end in eos_ids, f'{IM_END} ({im_end}) is not a generation stop id {sorted(eos_ids)}'
    ps = processor.image_processor.patch_size
    longest = Counter()
    for r in ds:
        enc = processor(text=[r['prompt']], images=r['images'], return_tensors='pt')
        for im, thw in zip(r['images'], enc['image_grid_thw']):
            w, h = im.size
            assert [int(x) for x in thw] == [1, h // ps, w // ps], f"{r['id']}: processor resized {im.size}"
        n_prompt = enc['input_ids'].shape[1]
        for side in ('chosen', 'rejected'):
            ids = tok(r[side], add_special_tokens=False)['input_ids']
            assert (ids[-1] == im_end) == r[side].endswith(IM_END)
            longest[f'{r["kind"]}/{side}'] = max(longest[f'{r["kind"]}/{side}'], n_prompt + len(ids))
    print(f'[check] {len(ds)} pairs: no second resize, {IM_END} is EOS; longest prompt+completion '
          + ', '.join(f'{k} {v}' for k, v in sorted(longest.items())))
    if max_length and max(longest.values()) > max_length:
        print(f'[check] WARNING: sequences longer than max_length {max_length} will lose their END '
              f'(keep_start truncation), i.e. the EOS of chosen')
    return max(longest.values())


def make_dpo_config(args, **overrides):
    from trl import DPOConfig
    kw = dict(
        output_dir=args.output_dir, seed=args.seed, report_to=args.report_to,
        model_init_kwargs={'dtype': 'bfloat16'}, bf16=True, gradient_checkpointing=True,
        learning_rate=args.learning_rate, lr_scheduler_type='cosine',
        warmup_steps=args.warmup_ratio,   # transformers 5: a float < 1 is a fraction of total steps
        num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        beta=args.beta,
        loss_type=['sigmoid', 'sft'] if args.sft_weight > 0 else ['sigmoid'],
        loss_weights=[1.0, args.sft_weight] if args.sft_weight > 0 else None,
        ld_alpha=args.ld_alpha, max_length=args.max_length,
        precompute_ref_log_probs=False,   # not supported for vision datasets in TRL 1.17
        remove_unused_columns=False, logging_steps=1,
        save_strategy='steps', save_steps=args.save_steps, save_total_limit=4,
        eval_strategy='steps' if args.val_pairs else 'no', eval_steps=args.save_steps if args.val_pairs else None,
    )
    return DPOConfig(**{**kw, **overrides})


def main(argv=None):
    args = parse_args(argv)
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, GenerationConfig
    import infer
    rev = args.model_revision if args.model_revision is not None else infer.MODEL_REVISION
    model_dir = args.model_path if os.path.isdir(args.model_path) else snapshot_download(args.model_path,
                                                                                          revision=rev or None)
    processor = AutoProcessor.from_pretrained(model_dir)
    processor.image_processor.size = {'shortest_edge': EVAL.min_pixels, 'longest_edge': EVAL.max_pixels}
    eos = GenerationConfig.from_pretrained(model_dir).eos_token_id
    eos_ids = set([eos] if isinstance(eos, int) else eos)

    pairs = [p for p in read_jsonl(args.pairs) if p['kind'] in args.kinds][:args.limit]
    print(f'[data] {len(pairs)} pairs: {dict(Counter(p["kind"] for p in pairs))}', flush=True)
    ds = build_dataset(pairs, processor, args.image_root)
    check(processor, ds, eos_ids, args.max_length)
    val = None
    if args.val_pairs:
        val = build_dataset([p for p in read_jsonl(args.val_pairs) if p['kind'] in args.kinds], processor,
                            args.image_root)
    if args.check_only:
        return

    import torch
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import DPOTrainer
    from train_ft import LORA_TARGET

    config = make_dpo_config(args)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGET,
                      bias='none', task_type='CAUSAL_LM')
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')

    class Bookkeeping(TrainerCallback):
        def on_train_begin(self, a, state, control, **kw):
            # resuming restores save/log/eval steps from the checkpoint's trainer_state.json (run1 GRPO lost
            # its 25/50/75 checkpoints that way); use this run's arguments instead
            state.save_steps, state.logging_steps = a.save_steps, a.logging_steps
            if a.eval_steps:
                state.eval_steps = a.eval_steps

        def on_log(self, a, state, control, logs=None, **kw):
            if logs:
                with open(metrics_path, 'a') as f:
                    f.write(json.dumps({'step': state.global_step, **logs,
                                        'max_mem_gb': torch.cuda.max_memory_allocated() / 2**30}) + '\n')

    trainer = DPOTrainer(model=model_dir, args=config, train_dataset=ds, eval_dataset=val,
                         processing_class=processor, peft_config=lora, callbacks=[Bookkeeping()])
    lora_mods = [n for n, _ in trainer.model.named_modules() if n.endswith('lora_A')]
    assert lora_mods and not any('visual' in n for n in lora_mods), 'LoRA must stay on the LLM decoder'
    with open(os.path.join(args.output_dir, 'run_args.json'), 'w') as f:
        json.dump(vars(args) | {'model_dir': model_dir, 'lora_target': LORA_TARGET, 'n_pairs': len(ds)}, f, indent=2)
    trainer.train(resume_from_checkpoint=any(d.startswith('checkpoint-') for d in os.listdir(args.output_dir)))
    trainer.save_model(os.path.join(args.output_dir, 'final_adapter'))


if __name__ == '__main__':
    main()
