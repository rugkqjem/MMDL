"""MMMU-Pro inference for Qwen3-VL with vLLM (test split, answers public). Same model / sampling / pixel settings,
vLLM engine and output format as infer.py, so score.py and run_mmmu_eval.sh work unchanged.

Settings (MMMU-Pro paper, arXiv:2409.02813): the official score is the mean of
  standard : 'standard (10 options)' config, question + up to 10 options as text, images as in MMMU
  vision  : 'vision' config, one screenshot/photo that contains the question and options, no text besides the instruction
Instructions are the official CoT prompts (MMMU-Benchmark/MMMU mmmu-pro/prompts.yaml, 'cot'). Option layout follows
the official construct_prompt() ("A. ..." lines). Images go before the text, as in infer.build_messages().

    python scripts/infer_mmmu_pro.py --setting standard --output outputs/pro_std/predictions.jsonl --data_root $HF_DATA
"""
import ast
import json
import os
import string
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer  # noqa: E402

COT = {
    'vision': "Write out the multiple-choice question in the image and then solve it. The last line of your response "
              "should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. "
              "Think step by step before answering.",
    'standard': "Answer the preceding multiple choice question. The last line of your response should be of the "
                "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. "
                "Think step by step before answering.",
}
CONFIG = {'standard': 'standard (10 options)', 'vision': 'vision'}


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    setting = 'standard'
    if '--setting' in argv:
        i = argv.index('--setting')
        setting = argv[i + 1]
        del argv[i:i + 2]
    if setting not in CONFIG:
        raise SystemExit(f'--setting must be one of {list(CONFIG)}')
    pro_rev = None
    if '--pro_revision' in argv:
        i = argv.index('--pro_revision')
        pro_rev = argv[i + 1] or None
        del argv[i:i + 2]
    args = infer.parse_args(argv)
    args.setting, args.pro_revision = setting, pro_rev
    return args


def build_messages(sample, setting, min_pixels, max_pixels):
    options = sample['options']
    options = ast.literal_eval(options) if isinstance(options, str) else list(options or [])
    if setting == 'vision':
        images = [sample['image']]
        prompt = COT['vision']
    else:
        images = [sample[f'image_{i}'] for i in range(1, 8) if sample.get(f'image_{i}') is not None]
        opts = '\n'.join(f'{k}. {o}' for k, o in zip(string.ascii_uppercase, options))
        prompt = f"{sample['question']}\n{opts}\n{COT['standard']}"
    content = [{'type': 'image', 'image': im, 'min_pixels': min_pixels, 'max_pixels': max_pixels} for im in images]
    content.append({'type': 'text', 'text': prompt})
    return [{'role': 'user', 'content': content}], prompt, options


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

    processor = AutoProcessor.from_pretrained(args.model_path, revision=args.model_revision or None)
    ds = load_dataset('MMMU/MMMU_Pro', CONFIG[args.setting], split='test', revision=args.pro_revision,
                      cache_dir=args.data_root)
    records, inputs = [], []
    for sample in ds:
        messages, prompt, options = build_messages(sample, args.setting, args.min_pixels, args.max_pixels)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _, video_kwargs = process_vision_info(
            messages, image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True, return_video_metadata=True)
        inputs.append({'prompt': text, 'multi_modal_data': {'image': image_inputs}, 'mm_processor_kwargs': video_kwargs})
        records.append({'id': sample['id'], 'subject': sample['subject'], 'question_type': 'multiple-choice',
                        'question': sample.get('question', ''), 'options': options, 'answer': sample['answer'],
                        'prompt': prompt, 'num_images': len(image_inputs),
                        'image_sizes': [list(im.size) for im in image_inputs]})
    print(f'prepared {len(inputs)} samples ({CONFIG[args.setting]})')
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
    meta = {'args': vars(args), 'dataset': f"MMMU/MMMU_Pro:{CONFIG[args.setting]}:test",
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
