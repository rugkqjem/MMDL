"""GRPO for Qwen3-VL-4B-Instruct with TRL GRPOTrainer + LoRA, rewarded by the eval's own scorer.

Same as the MMMU eval, so the reward optimizes the reported metric:
  - prompt: the `prompt` field of data/rl/*.jsonl came from infer.build_messages(); images go before the text
  - image resolution: every image is resized here with qwen_vl_utils exactly as infer.prepare_sample() does
    (infer.py min/max pixels), so the trainer's processor and vLLM receive eval-sized images and leave them as is
  - scoring: rl_reward.py (extract_final_answer + score_qwen) with a truncation penalty
  - weights: the pinned revision (infer.MODEL_REVISION), one local snapshot shared by the trainer and vLLM
--check_only verifies prompt and image parity with the real processor before any GPU work.

Deliberately different from the eval:
  - sampling (--sampling train, default): temperature 1.0, top_p 1.0, top_k 0, no presence/repetition penalty.
    The trainer recomputes log-probs with a plain softmax, so filtered/penalized sampling makes rollouts come from a
    different distribution, and a presence penalty would hide the loops we want the model to stop on its own.
    --sampling eval keeps the eval's temperature/top_p/top_k (presence penalty still off), for comparison.
  - max response 8192 tokens (eval 16384), the length the rollouts were filtered with.
LoRA: rank 16 on every linear layer of the LLM decoder (train_ft.LORA_TARGET); ViT and mergers frozen.
"LoRA Without Regret" (Thinking Machines, 2025; TRL docs): all layers beat attention-only, RL needs low rank,
lr about 10x full fine-tuning, effective batch below 32.

  # 1) on the server, CPU only (processor download): data, prompt parity, image grid, prompt lengths, reward wiring
  python scripts/train_grpo.py --check_only
  # 2) pilot, 20 optimizer steps: read outputs/grpo/pilot/metrics.jsonl
  python scripts/train_grpo.py --output_dir outputs/grpo/pilot --max_steps 20
  # 3) full run; then merge the adapter and evaluate with the eval env (venv, not venv_rl)
  python scripts/train_grpo.py --output_dir outputs/grpo/run1
  python scripts/merge_lora.py --adapter outputs/grpo/run1/checkpoint-XXX --out outputs/grpo/run1/merged
Env: the eval env plus requirements.rl.txt, installed with requirements.lock.txt as a constraint so torch / vLLM /
transformers stay exactly the eval versions (TRL 1.17 supports vLLM 0.22-0.33; the lock has 0.30).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer  # noqa: E402
from rl_reward import DEFAULT_TRUNC_PENALTY, make_reward_func  # noqa: E402

EVAL = infer.parse_args(['--output', os.devnull])  # eval defaults: pixels, sampling recipe, model revision


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--train', default='data/rl/train_final.jsonl')
    p.add_argument('--val', default='data/rl/train_val_final.jsonl', help='used with --eval_steps > 0')
    p.add_argument('--image_root', default='data/virl39k_raw')
    p.add_argument('--output_dir', default='outputs/grpo/run1')
    p.add_argument('--model_path', default=EVAL.model_path)
    p.add_argument('--model_revision', default=EVAL.model_revision)
    p.add_argument('--check_only', action='store_true', help='CPU checks, no training')
    p.add_argument('--limit', type=int, default=None, help='first N training rows (smoke tests)')
    # reward / generation
    p.add_argument('--trunc_penalty', type=float, default=DEFAULT_TRUNC_PENALTY)
    p.add_argument('--max_completion_length', type=int, default=8192)
    p.add_argument('--num_generations', type=int, default=8)
    p.add_argument('--sampling', choices=['train', 'eval'], default='train')
    # optimization
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=1e-5)
    p.add_argument('--warmup_steps', type=float, default=5)
    p.add_argument('--num_train_epochs', type=float, default=2)
    p.add_argument('--max_steps', type=int, default=-1)
    p.add_argument('--per_device_train_batch_size', type=int, default=2, help='completions per micro-batch')
    p.add_argument('--gradient_accumulation_steps', type=int, default=16, help='2 x 16 = 32 completions/step')
    p.add_argument('--generation_batch_size', type=int, default=128,
                   help='completions per vLLM call (16 questions x 8); larger keeps vLLM busy past long stragglers')
    p.add_argument('--loss_type', default='dapo')
    # vLLM (colocated on the training GPU)
    p.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.35)
    p.add_argument('--vllm_max_model_length', type=int, default=None, help='default: longest prompt + max_completion')
    p.add_argument('--no_sleep_mode', action='store_true', help='keep vLLM resident during the optimizer step')
    # bookkeeping
    p.add_argument('--save_steps', type=int, default=25)
    p.add_argument('--eval_steps', type=int, default=0, help='0 = off; evaluate merged checkpoints with infer.py')
    p.add_argument('--report_to', default='none')
    p.add_argument('--seed', type=int, default=EVAL.seed)
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- data
def load_rows(path, limit=None):
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def eval_messages(images, prompt, min_pixels, max_pixels):
    """The message layout of infer.build_messages(): every image first, then the text."""
    content = [{'type': 'image', 'image': im, 'min_pixels': min_pixels, 'max_pixels': max_pixels} for im in images]
    return [{'role': 'user', 'content': content + [{'type': 'text', 'text': prompt}]}]


def resize_like_eval(row, image_root, patch_size, min_pixels=EVAL.min_pixels, max_pixels=EVAL.max_pixels):
    """Open the row's images (RGB, as rl_rollout_filter.to_sample) and resize them with the eval's
    process_vision_info call (infer.prepare_sample)."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    paths = row['images'] if isinstance(row['images'], list) else [row['images']]
    imgs = [Image.open(p if os.path.isabs(p) else os.path.join(image_root, p)).convert('RGB') for p in paths]
    image_inputs, _, _ = process_vision_info(eval_messages(imgs, row['prompt'], min_pixels, max_pixels),
                                             image_patch_size=patch_size,
                                             return_video_kwargs=True, return_video_metadata=True)
    return image_inputs


def to_trl_row(row, images):
    """TRL conversational VLM row: {'type': 'image'} placeholders + `images`; answer/options reach the reward."""
    if row['problem'] != '<image>' * len(images) + row['prompt']:
        raise ValueError(f"{row['id']}: image count does not match the <image> tags in `problem`")
    content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': row['prompt']}]
    return {'id': row['id'], 'prompt': [{'role': 'user', 'content': content}], 'images': images,
            'answer': row['answer'], 'options': row['options'], 'category': row.get('category', '')}


def build_dataset(rows, image_root, patch_size):
    from datasets import Dataset
    t0, out = time.time(), []
    for i, r in enumerate(rows):
        out.append(to_trl_row(r, resize_like_eval(r, image_root, patch_size)))
        if (i + 1) % 200 == 0:
            print(f'[data] {i + 1}/{len(rows)} resized ({time.time() - t0:.0f}s)', flush=True)
    return Dataset.from_list(out)


# ----------------------------------------------------------------------------- checks
def check_parity(processor, rows, ds):
    """Chat-template text identical to the eval's, processor does not resize again, prompt token lengths."""
    ps, merge = processor.image_processor.patch_size, processor.image_processor.merge_size
    longest = 0
    for i, r in enumerate(rows):
        imgs = ds[i]['images']
        trl_text = processor.apply_chat_template(ds[i]['prompt'], tokenize=False, add_generation_prompt=True)
        ev_text = processor.apply_chat_template(eval_messages(['x'] * len(imgs), r['prompt'], 1, 2),
                                                tokenize=False, add_generation_prompt=True)
        assert trl_text == ev_text, f"{r['id']}: chat template differs from the eval prompt"
        enc = processor(text=[trl_text], images=imgs, return_tensors='pt')
        for im, thw in zip(imgs, enc['image_grid_thw']):
            w, h = im.size
            assert [int(x) for x in thw] == [1, h // ps, w // ps], f"{r['id']}: processor resized {im.size} -> grid {thw.tolist()}"
        longest = max(longest, enc['input_ids'].shape[1])
    print(f'[check] {len(rows)} rows: chat template == eval, no second resize '
          f'(patch {ps}, merge {merge}), longest prompt {longest} tokens')
    return longest


def check_reward(reward, ds, max_len, eos_ids):
    a, o = ds[0]['answer'], ds[0]['options']
    eos = sorted(eos_ids)[0]
    got = reward(completions=[f'The answer is ({a}).', 'The answer is (Z).', f'The answer is ({a}).'],
                 completion_ids=[[1, eos], [1, eos], [1] * max_len], answer=[a] * 3, options=[o] * 3)
    assert got[0] == 1.0 and got[1] == 0.0 and got[2] < 0, got
    print(f'[check] reward wiring: correct {got[0]}, wrong {got[1]}, truncated {got[2]}')


# ----------------------------------------------------------------------------- configs
def sampling_params(args):
    if args.sampling == 'train':
        return dict(temperature=1.0, top_p=1.0, top_k=0)
    return dict(temperature=EVAL.temperature, top_p=EVAL.top_p, top_k=EVAL.top_k)


def make_grpo_config(args, max_model_len, with_eval=False, **overrides):
    """overrides: only for CPU tests (bf16=False, use_cpu=True)."""
    from trl import GRPOConfig
    sampling = sampling_params(args)
    kw = dict(
        output_dir=args.output_dir, seed=args.seed, report_to=args.report_to,
        model_init_kwargs={'dtype': 'bfloat16'}, bf16=True, gradient_checkpointing=True,
        learning_rate=args.learning_rate, lr_scheduler_type='constant_with_warmup', warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=args.generation_batch_size, num_generations=args.num_generations,
        max_completion_length=args.max_completion_length, repetition_penalty=1.0, **sampling,
        # GRPO objective: no KL, token-level (DAPO) normalization, group std scaling
        beta=0.0, loss_type=args.loss_type, scale_rewards='group',
        mask_truncated_completions=False,  # True would drop truncated samples and with them the loop signal
        use_vllm=True, vllm_mode='colocate', vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=max_model_len, vllm_enable_sleep_mode=not args.no_sleep_mode,
        remove_unused_columns=False, logging_steps=1, log_completions=True, num_completions_to_print=1,
        save_strategy='steps', save_steps=args.save_steps, save_total_limit=3,
        eval_strategy='steps' if with_eval else 'no', eval_steps=args.eval_steps or None,
    )
    return GRPOConfig(**{**kw, **overrides}), sampling


def make_lora_config(args):
    from peft import LoraConfig
    from train_ft import LORA_TARGET
    return LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGET,
                      bias='none', task_type='CAUSAL_LM')


# ----------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, GenerationConfig
    model_dir = args.model_path if os.path.isdir(args.model_path) else snapshot_download(
        args.model_path, revision=args.model_revision or None)
    processor = AutoProcessor.from_pretrained(model_dir)
    processor.image_processor.size = {'shortest_edge': EVAL.min_pixels, 'longest_edge': EVAL.max_pixels}
    processor.tokenizer.padding_side = 'left'  # GRPOTrainer requirement
    eos_ids = GenerationConfig.from_pretrained(model_dir).eos_token_id
    eos_ids = set([eos_ids] if isinstance(eos_ids, int) else eos_ids)
    reward = make_reward_func(args.max_completion_length, eos_ids, args.trunc_penalty)

    rows = load_rows(args.train, args.limit)
    ds = build_dataset(rows, args.image_root, processor.image_processor.patch_size)
    longest = check_parity(processor, rows, ds)
    check_reward(reward, ds, args.max_completion_length, eos_ids)
    max_model_len = args.vllm_max_model_length or longest + args.max_completion_length + 64
    if longest + args.max_completion_length > max_model_len:
        sys.exit(f'vllm_max_model_length {max_model_len} < longest prompt {longest} + {args.max_completion_length}')
    if args.check_only:
        print(f'[check] OK. vllm_max_model_length would be {max_model_len}; eos ids {sorted(eos_ids)}')
        return

    import torch
    from transformers import TrainerCallback
    from trl import GRPOTrainer
    from train_ft import LORA_TARGET

    val = None
    if args.eval_steps:
        val = build_dataset(load_rows(args.val), args.image_root, processor.image_processor.patch_size)
    config, sampling = make_grpo_config(args, max_model_len, with_eval=val is not None)
    lora = make_lora_config(args)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')
    keys = ('reward', 'reward_std', 'frac_reward_zero_std', 'completions/clipped_ratio',
            'completions/mean_length', 'reward_parts/correct', 'reward_parts/truncated', 'step_time')

    class MetricsLog(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if not logs:
                return
            rec = {'step': state.global_step, **logs, 'max_mem_gb': torch.cuda.max_memory_allocated() / 2**30}
            with open(metrics_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            short = {k: round(v, 3) for k, v in logs.items() if k in keys}
            print(f'[step {state.global_step}] {short}', flush=True)

    trainer = GRPOTrainer(model=model_dir, reward_funcs=reward, args=config, train_dataset=ds, eval_dataset=val,
                          processing_class=processor, peft_config=lora, callbacks=[MetricsLog()])
    lora_mods = [n for n, _ in trainer.model.named_modules() if n.endswith('lora_A')]
    assert lora_mods and not any('visual' in n for n in lora_mods), 'LoRA must stay on the LLM decoder'
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f'[lora] {len(lora_mods)} adapted linears, {n_train:,} trainable params; sampling {sampling}; '
          f'trunc_penalty {args.trunc_penalty}; vllm_max_model_length {max_model_len}', flush=True)
    with open(os.path.join(args.output_dir, 'run_args.json'), 'w') as f:
        json.dump(vars(args) | {'model_dir': model_dir, 'sampling_params': sampling, 'eos_ids': sorted(eos_ids),
                                'vllm_max_model_length': max_model_len, 'lora_target': LORA_TARGET}, f, indent=2)
    trainer.train(resume_from_checkpoint=any(d.startswith('checkpoint-') for d in os.listdir(args.output_dir)))
    trainer.save_model(os.path.join(args.output_dir, 'final_adapter'))


if __name__ == '__main__':
    main()
