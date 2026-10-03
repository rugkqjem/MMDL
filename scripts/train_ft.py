"""Fine-tune Qwen3-VL: DeepStack mergers (full) and/or the LLM (LoRA), with the vision encoder frozen.

Qwen3-VL taps the outputs of a few ViT blocks (`config.vision_config.deepstack_visual_indexes`), passes each through
its own merger (`model.model.visual.deepstack_merger_list`) and adds the result to the visual-token hidden states of
the first LLM layers (transformers `Qwen3VLTextModel._deepstack_process`). Those mergers are the only DeepStack-specific
weights, so with the ViT frozen they are what --tune_deepstack trains. --lora_llm adds LoRA adapters to the LLM's
attention / MLP projections. Ablation arms (scripts/run_ablation.sh): deepstack only, LoRA only, both.

Written against transformers 5.17 / peft 0.21 (requirements.lock.txt): the vision tower lives at
`model.model.visual` (4.57's `model.visual` property is gone).

Training data: JSON list or JSONL in the QwenLM/Qwen3-VL qwen-vl-finetune format, one sample per row:
    {"image": "a.jpg" | ["a.jpg", "b.jpg"],
     "conversations": [{"from": "human", "value": "<image>\\nQuestion ..."}, {"from": "gpt", "value": "Answer ..."}]}
Each `<image>` in the text is replaced by the next image; images without a placeholder go before the first user
text (the layout infer.py uses). Only assistant turns are supervised. scripts/prepare_mmmu_sft.py writes MMMU in
this format.

The output dir is a full bf16 checkpoint (LoRA merged into the weights) + processor, so it is evaluated as is:
    bash scripts/run_mmmu_eval.sh --model_path <output_dir> --model_revision '' ...
"""
import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, set_seed

MODEL_REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
ROLES = {'human': 'user', 'user': 'user', 'gpt': 'assistant', 'assistant': 'assistant', 'system': 'system'}
IGNORE_INDEX = -100
SEQ_KEYS = ('input_ids', 'attention_mask', 'mm_token_type_ids', 'token_type_ids')
# Every linear projection of the LLM decoder layers; the ViT and its mergers never get adapters.
LORA_TARGET = r'.*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='Qwen/Qwen3-VL-4B-Instruct', help='HF repo id or local checkpoint dir')
    p.add_argument('--model_revision', default=MODEL_REVISION, help="ignored for local dirs; pass '' to disable")
    p.add_argument('--train_data', required=True, help='JSON / JSONL in qwen-vl-finetune format')
    p.add_argument('--image_root', default='', help='prefix for relative image paths')
    p.add_argument('--output_dir', required=True)
    # what to train (vision encoder always frozen)
    p.add_argument('--tune_deepstack', action='store_true', help='fully train visual.deepstack_merger_list')
    p.add_argument('--tune_merger', action='store_true', help='fully train visual.merger (off in the ablation)')
    p.add_argument('--lora_llm', action='store_true', help='LoRA on the LLM, merged into the weights at the end')
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--lora_target', default=LORA_TARGET, help='regex over module names')
    # Same bounds as infer.py so training sees images at the resolution they are evaluated at.
    p.add_argument('--min_pixels', type=int, default=1280 * 28 * 28)
    p.add_argument('--max_pixels', type=int, default=5120 * 28 * 28)
    p.add_argument('--max_length', type=int, default=8192, help='samples longer than this are skipped')
    p.add_argument('--learning_rate', type=float, default=1e-5, help='for the DeepStack / merger weights')
    p.add_argument('--lora_learning_rate', type=float, default=1e-4, help='for the LoRA adapters')
    p.add_argument('--num_train_epochs', type=float, default=1.0)
    p.add_argument('--max_steps', type=int, default=-1, help='overrides num_train_epochs when > 0')
    p.add_argument('--per_device_train_batch_size', type=int, default=1)
    p.add_argument('--gradient_accumulation_steps', type=int, default=8)
    p.add_argument('--warmup_steps', type=float, default=0.03, help='int = steps, float in [0, 1) = ratio of total')
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--save_steps', type=int, default=500)
    p.add_argument('--save_strategy', default='steps', choices=['steps', 'epoch', 'no'],
                   help="Trainer checkpoints (resumable); the final model is saved regardless. 'no' for smoke runs")
    p.add_argument('--logging_steps', type=int, default=10)
    p.add_argument('--dataloader_num_workers', type=int, default=4)
    p.add_argument('--gradient_checkpointing', action='store_true')
    p.add_argument('--attn_implementation', default='sdpa', help="'flash_attention_2' if installed")
    p.add_argument('--deepspeed', default=None, help='DeepSpeed config JSON, passed to TrainingArguments')
    p.add_argument('--seed', type=int, default=3407)
    # auto = bf16 when the GPU has native bf16 (A100), else fp16 (T4/V100, for run_ablation.sh --smoke only:
    # frozen weights round-trip through fp16, so such checkpoints are not for reported results). Saved as bf16 either way.
    p.add_argument('--dtype', default='auto', choices=['auto', 'bf16', 'fp16'])
    p.add_argument('--use_cpu', action='store_true', help='smoke tests only (scripts/test_train.py)')
    args = p.parse_args(argv)
    if not (args.tune_deepstack or args.tune_merger or args.lora_llm):
        p.error('nothing to train: pass --tune_deepstack and/or --lora_llm')
    return args


def resolve_dtype(name, use_cpu):
    if name == 'auto':
        native_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
        name = 'bf16' if use_cpu or native_bf16 else 'fp16'
    return name


def freeze_and_unfreeze(model, args):
    """Freeze everything, then unfreeze the selected full-tuned modules. LoRA adapters are added afterwards."""
    visual = model.model.visual
    for p in model.parameters():
        p.requires_grad = False
    modules = ([visual.deepstack_merger_list] if args.tune_deepstack else []) + \
              ([visual.merger] if args.tune_merger else [])
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True
    return modules


def add_lora(model, args, full_tuned):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        target_modules=args.lora_target, bias='none')
    model = get_peft_model(model, config)
    # get_peft_model freezes every non-adapter weight; turn the full-tuned modules back on.
    for m in full_tuned:
        for p in m.parameters():
            p.requires_grad = True
    return model


def load_rows(path):
    with open(path) as f:
        if path.endswith('.jsonl'):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def to_messages(row, image_root):
    """qwen-vl-finetune row -> chat-template messages with PIL images inlined at their <image> placeholders."""
    images = row.get('image') or []
    if isinstance(images, str):
        images = [images]
    images = [Image.open(os.path.join(image_root, x)).convert('RGB') for x in images]
    n_placeholders = sum(t['value'].count('<image>') for t in row['conversations'])
    if n_placeholders not in (0, len(images)):
        raise ValueError(f'{n_placeholders} <image> placeholders but {len(images)} images')

    messages, it = [], iter(images)
    for turn in row['conversations']:
        role = ROLES[turn['from']]
        content = []
        if role == 'user' and n_placeholders == 0 and not any(m['role'] == 'user' for m in messages):
            content += [{'type': 'image', 'image': img} for img in images]
        for i, chunk in enumerate(turn['value'].split('<image>')):
            if i > 0:
                content.append({'type': 'image', 'image': next(it)})
            if chunk.strip():
                content.append({'type': 'text', 'text': chunk})
        messages.append({'role': role, 'content': content})
    return messages


def assistant_labels(input_ids, tokenizer):
    """Copy input_ids into labels only inside assistant turns: after `<|im_start|>assistant\\n` up to and including
    the closing `<|im_end|>`, so the model learns to stop."""
    im_start = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    header = tokenizer.encode('assistant\n', add_special_tokens=False)
    ids = input_ids.tolist()
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    i = 0
    while i < len(ids):
        if ids[i] == im_start and ids[i + 1:i + 1 + len(header)] == header:
            start = i + 1 + len(header)
            end = start
            while end < len(ids) and ids[end] != im_end:
                end += 1
            labels[start:end + 1] = input_ids[start:end + 1]
            i = end
        i += 1
    return labels


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows, processor, image_root, max_length):
        self.rows, self.processor, self.image_root, self.max_length = rows, processor, image_root, max_length

    def __len__(self):
        return len(self.rows)

    def encode(self, idx):
        messages = to_messages(self.rows[idx], self.image_root)
        enc = self.processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors='pt')
        enc = {k: v[0] if k in SEQ_KEYS else v for k, v in enc.items()}
        enc['labels'] = assistant_labels(enc['input_ids'], self.processor.tokenizer)
        return enc

    def __getitem__(self, idx):
        # Skip over-long samples instead of truncating: a cut can split an image's tokens from its pixel patches.
        for k in range(len(self.rows)):
            enc = self.encode((idx + k) % len(self.rows))
            if len(enc['input_ids']) <= self.max_length:
                return enc
        raise ValueError(f'every sample is longer than --max_length {self.max_length}')


def make_collator(pad_token_id):
    pad_values = {'input_ids': pad_token_id, 'labels': IGNORE_INDEX}

    def collate(batch):
        out = {}
        for key in batch[0]:
            if key in SEQ_KEYS or key == 'labels':
                # right padding, matching the processor's attention_mask convention
                out[key] = torch.nn.utils.rnn.pad_sequence(
                    [b[key] for b in batch], batch_first=True, padding_value=pad_values.get(key, 0))
            else:  # pixel_values / image_grid_thw are already flattened across images
                out[key] = torch.cat([b[key] for b in batch], dim=0)
        return out

    return collate


class GroupLRTrainer(Trainer):
    """Separate learning rates for the full-tuned weights and the LoRA adapters."""

    def __init__(self, *a, lora_lr=None, **kw):
        super().__init__(*a, **kw)
        self.lora_lr = lora_lr

    def create_optimizer(self, model=None):
        if self.optimizer is None:
            params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            groups = [{'params': [p for n, p in params if 'lora_' not in n], 'lr': self.args.learning_rate},
                      {'params': [p for n, p in params if 'lora_' in n], 'lr': self.lora_lr}]
            groups = [g for g in groups if g['params']]
            for g in groups:
                g['weight_decay'] = self.args.weight_decay
            self.optimizer = torch.optim.AdamW(groups, betas=(self.args.adam_beta1, self.args.adam_beta2),
                                               eps=self.args.adam_epsilon)
        return self.optimizer


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    revision = None if os.path.isdir(args.model_path) else (args.model_revision or None)
    dtype = resolve_dtype(args.dtype, args.use_cpu)
    print(f'compute dtype: {dtype}')

    processor = AutoProcessor.from_pretrained(args.model_path, revision=revision)
    processor.image_processor.size = {'shortest_edge': args.min_pixels, 'longest_edge': args.max_pixels}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, revision=revision, dtype=torch.bfloat16 if dtype == 'bf16' else torch.float16,
        attn_implementation=args.attn_implementation)
    model.config.use_cache = False

    full_tuned = freeze_and_unfreeze(model, args)
    if args.lora_llm:
        model = add_lora(model, args, full_tuned)
    # fp32 master copies for the trained weights (tiny lr updates vanish in bf16/fp16); cast back before the final save.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f'trainable: {n_train:,} / {n_all:,} params ({100 * n_train / n_all:.3f}%) in {len(trainable)} tensors')

    dataset = SFTDataset(load_rows(args.train_data), processor, args.image_root, args.max_length)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type='cosine',
        bf16=dtype == 'bf16',
        fp16=dtype == 'fp16',  # trained weights are fp32, so the fp16 GradScaler can unscale them
        use_cpu=args.use_cpu,
        logging_steps=args.logging_steps,
        # With LoRA, Trainer checkpoints hold only the adapters (no DeepStack weights); the final save below is the
        # full merged model, so treat intermediate checkpoints as LoRA-only snapshots.
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=2,
        # Non-reentrant checkpointing: the frozen layers' inputs do not require grad, the DeepStack sums do.
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        deepspeed=args.deepspeed,
        seed=args.seed,
        report_to='none',
    )
    trainer = GroupLRTrainer(model=model, args=training_args, train_dataset=dataset,
                             data_collator=make_collator(processor.tokenizer.pad_token_id),
                             lora_lr=args.lora_learning_rate)
    trainer.train()

    if trainer.is_world_process_zero():
        model = trainer.model
        if args.lora_llm:
            model = model.merge_and_unload()
        model.to(torch.bfloat16)
        model.config.use_cache = True
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, 'train_args.json'), 'w') as f:
            json.dump(vars(args) | {'trainable_tensors': trainable}, f, indent=2)


if __name__ == '__main__':
    main()
