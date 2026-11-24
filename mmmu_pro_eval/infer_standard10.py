"""MMMU-Pro 'standard (10 options)' inference (test split, 1,730 questions) for one model with vLLM.

Builds on scripts/infer.py (model loading, sampling, pixel settings, output format) and
scripts/infer_mmmu_pro.py (official CoT instruction, "A. ..." option lines), so the output JSONL is scored by
scripts/score.py unchanged.

What this script adds: --image_layout
  interleave (default): every '<image n>' placeholder in the question/options is replaced by image_n at that
                        position, as in the official MMMU-Pro inference code. An image used twice is inserted
                        once (first occurrence); later mentions stay as text. Images that no placeholder
                        mentions go in front of the text.
  front               : all images before the text, placeholders left as text (= scripts/infer_mmmu_pro.py and
                        the repo's MMMU-val layout, Qwen run_mmmu.py style).

    python mmmu_pro_eval/infer_standard10.py --model_path Qwen/Qwen3-VL-4B-Instruct \
        --output outputs/mmmu_pro_std10/baseline/predictions.jsonl --data_root $HF_DATA
"""
import ast
import json
import os
import re
import string
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import infer  # noqa: E402
from infer_mmmu_pro import COT  # noqa: E402

CONFIG = 'standard (10 options)'
PLACEHOLDER = re.compile(r'<image (\d+)>')


def pop_arg(argv, name, default):
    if name in argv:
        i = argv.index(name)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    layout = pop_arg(argv, '--image_layout', 'interleave')
    if layout not in ('interleave', 'front'):
        raise SystemExit('--image_layout must be interleave or front')
    pro_rev = pop_arg(argv, '--pro_revision', None) or None
    limit = int(pop_arg(argv, '--limit', 0))
    args = infer.parse_args(argv)  # --subjects / --split / --dataset_revision are MMMU-only and ignored here
    args.image_layout, args.pro_revision, args.limit = layout, pro_rev, limit
    return args


def build_prompt(sample):
    options = sample['options']
    options = ast.literal_eval(options) if isinstance(options, str) else list(options or [])
    opts = '\n'.join(f'{k}. {o}' for k, o in zip(string.ascii_uppercase, options))
    return f"{sample['question']}\n{opts}\n{COT['standard']}", options


def build_messages(sample, layout, min_pixels, max_pixels):
    """-> (messages, prompt text, options, image order as 1-based indices)."""
    prompt, options = build_prompt(sample)
    avail = {i: sample[f'image_{i}'] for i in range(1, 8) if sample.get(f'image_{i}') is not None}
    img = lambda i: {'type': 'image', 'image': avail[i], 'min_pixels': min_pixels, 'max_pixels': max_pixels}  # noqa: E731

    if layout == 'front':
        order = sorted(avail)
        content = [img(i) for i in order] + [{'type': 'text', 'text': prompt}]
        return [{'role': 'user', 'content': content}], prompt, options, order

    body, used, pos = [], [], 0
    for m in PLACEHOLDER.finditer(prompt):
        i = int(m.group(1))
        if i not in avail or i in used:
            continue  # missing image or already inserted: keep the placeholder as text
        if m.start() > pos:
            body.append({'type': 'text', 'text': prompt[pos:m.start()]})
        body.append(img(i))
        used.append(i)
        pos = m.end()
    if pos < len(prompt):
        body.append({'type': 'text', 'text': prompt[pos:]})
    lead = [i for i in sorted(avail) if i not in used]
    content = [img(i) for i in lead] + body
    return [{'role': 'user', 'content': content}], prompt, options, lead + used


def main():
    args = parse_args()
    t0 = time.time()
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    import torch
    import transformers
    import vllm
    from datasets import load_dataset
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    revision = args.model_revision or None
    processor = AutoProcessor.from_pretrained(args.model_path, revision=revision)
    ds = load_dataset('MMMU/MMMU_Pro', CONFIG, split='test', revision=args.pro_revision, cache_dir=args.data_root)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    records, inputs = [], []
    for sample in ds:
        messages, prompt, options, order = build_messages(sample, args.image_layout, args.min_pixels, args.max_pixels)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _, video_kwargs = process_vision_info(
            messages, image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True, return_video_metadata=True)
        inputs.append({'prompt': text, 'multi_modal_data': {'image': image_inputs}, 'mm_processor_kwargs': video_kwargs})
        records.append({'id': sample['id'], 'subject': sample['subject'], 'question_type': 'multiple-choice',
                        'question': sample['question'], 'options': options, 'answer': sample['answer'],
                        'prompt': prompt, 'num_images': len(image_inputs), 'image_order': order,
                        'image_sizes': [list(im.size) for im in image_inputs]})
    print(f'prepared {len(inputs)} samples ({CONFIG}, layout={args.image_layout})')
    bad = [r['id'] for r in records if r['answer'] not in string.ascii_uppercase[:len(r['options'])]]
    if bad:
        raise SystemExit(f'{len(bad)} samples have an answer outside their options, e.g. {bad[:3]}')

    llm = infer.make_llm(args)
    vram = infer.PeakVRAM()
    vram.start()
    t_gen = time.time()
    outputs = llm.generate(inputs, sampling_params=infer.make_sampling(args))
    gen_sec = time.time() - t_gen
    vram.stop.set()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        for rec, out in zip(records, outputs):
            o = out.outputs[0]
            rec.update(response=o.text, finish_reason=o.finish_reason, num_output_tokens=len(o.token_ids))
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    meta = {'args': vars(args), 'dataset': f'MMMU/MMMU_Pro:{CONFIG}:test',
            'versions': {'vllm': vllm.__version__, 'transformers': transformers.__version__, 'torch': torch.__version__},
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'num_samples': len(records), 'num_truncated': sum(r['finish_reason'] == 'length' for r in records),
            'generate_seconds': round(gen_sec, 1), 'total_seconds': round(time.time() - t0, 1),
            'peak_vram_mib_nvidia_smi': vram.peak_mib}
    with open(os.path.splitext(args.output)[0] + '_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
