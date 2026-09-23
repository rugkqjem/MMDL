"""MMMU-val inference for Qwen3-VL with vLLM. Writes one raw-response JSONL; scoring is done by score.py.

Prompt / image handling / sampling follow QwenLM/Qwen3-VL evaluation/mmmu/run_mmmu.py
(@96588727e44c78b25ba03ea03b8e12f7e64fd0da); only data loading is swapped to the pinned HF MMMU/MMMU dataset.
"""
import argparse
import ast
import json
import os
import string
import subprocess
import threading
import time

SUBJECTS = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
    'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
    'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage',
    'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy',
    'Physics', 'Psychology', 'Public_Health', 'Sociology',
]
MODEL_REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
DATASET_REVISION = '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='Qwen/Qwen3-VL-4B-Instruct', help='HF repo id or local checkpoint dir')
    p.add_argument('--model_revision', default=MODEL_REVISION, help="ignored for local dirs; pass '' to disable")
    p.add_argument('--data_root', default=None, help='HF datasets cache dir for MMMU (default: HF cache)')
    p.add_argument('--dataset_revision', default=DATASET_REVISION)
    p.add_argument('--output', required=True, help='output JSONL path')
    p.add_argument('--subjects', nargs='+', default=SUBJECTS, help='subset for smoke tests; default all 30')
    # "VL" recipe from the pinned model card (huggingface.co/Qwen/Qwen3-VL-4B-Instruct @ebb281ec, README
    # "Generation Hyperparameters"; same values in its generation_config.json). The card's "Text" recipe
    # (1.0/1.0/40/2.0) is for text-only benchmarks. The card gives no seed; 3407 is from the GitHub README.
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=20)
    p.add_argument('--repetition_penalty', type=float, default=1.0)
    p.add_argument('--presence_penalty', type=float, default=1.5)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--max_new_tokens', type=int, default=16384)  # model card VL recipe out_seq_length
    # Qwen run_mmmu.py build_mmmu_prompt()
    p.add_argument('--min_pixels', type=int, default=1280 * 28 * 28)
    p.add_argument('--max_pixels', type=int, default=5120 * 28 * 28)
    p.add_argument('--max_model_len', type=int, default=32768)
    p.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    return p.parse_args()


def build_messages(sample, min_pixels, max_pixels):
    """Same text/image layout as Qwen run_mmmu.py build_mmmu_prompt(), fed from an HF MMMU row."""
    options = ast.literal_eval(sample['options']) if sample['options'] else []
    prompt = f"Question: {sample['question']}\n"
    if options:
        prompt += 'Options:\n'
        for key, item in zip(string.ascii_uppercase, options):
            prompt += f'{key}. {item}\n'
        prompt += 'Please select the correct answer from the options above. \n'
    prompt = prompt.rstrip()

    content = [
        {'type': 'image', 'image': sample[f'image_{i}'], 'min_pixels': min_pixels, 'max_pixels': max_pixels}
        for i in range(1, 8) if sample.get(f'image_{i}') is not None
    ]
    content.append({'type': 'text', 'text': prompt})
    return [{'role': 'user', 'content': content}], prompt, options


class PeakVRAM(threading.Thread):
    """Polls nvidia-smi; vLLM runs the engine in a subprocess so torch.cuda stats in this process are useless."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mib, self.stop = 0, threading.Event()

    def run(self):
        while not self.stop.wait(1.0):
            try:
                out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True).stdout
                self.peak_mib = max([self.peak_mib] + [int(x) for x in out.split()])
            except (OSError, ValueError):
                return


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
    from vllm import LLM, SamplingParams

    revision = args.model_revision or None
    processor = AutoProcessor.from_pretrained(args.model_path, revision=revision)

    records, inputs = [], []
    for subject in args.subjects:
        ds = load_dataset('MMMU/MMMU', subject, split='validation',
                          revision=args.dataset_revision, cache_dir=args.data_root)
        for sample in ds:
            messages, prompt, options = build_messages(sample, args.min_pixels, args.max_pixels)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, image_patch_size=processor.image_processor.patch_size,
                return_video_kwargs=True, return_video_metadata=True)
            inputs.append({'prompt': text, 'multi_modal_data': {'image': image_inputs},
                           'mm_processor_kwargs': video_kwargs})
            records.append({
                'id': sample['id'], 'subject': subject, 'question_type': sample['question_type'],
                'question': sample['question'], 'options': options, 'answer': sample['answer'],
                'prompt': prompt, 'num_images': len(image_inputs),
                'image_sizes': [list(im.size) for im in image_inputs],
            })
    print(f'prepared {len(inputs)} samples')

    llm = LLM(model=args.model_path, revision=revision, tokenizer_revision=revision, dtype='bfloat16',
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              limit_mm_per_prompt={'image': 7}, seed=args.seed)
    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                              repetition_penalty=args.repetition_penalty, presence_penalty=args.presence_penalty,
                              max_tokens=args.max_new_tokens, seed=args.seed)

    vram = PeakVRAM()
    vram.start()
    t_gen = time.time()
    outputs = llm.generate(inputs, sampling_params=sampling)  # all requests at once -> continuous batching
    gen_sec = time.time() - t_gen
    vram.stop.set()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        for rec, out in zip(records, outputs):
            o = out.outputs[0]
            rec.update(response=o.text, finish_reason=o.finish_reason, num_output_tokens=len(o.token_ids))
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    meta = {
        'args': vars(args),
        'versions': {'vllm': vllm.__version__, 'transformers': transformers.__version__, 'torch': torch.__version__},
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'num_samples': len(records),
        'num_truncated': sum(r['finish_reason'] == 'length' for r in records),
        'generate_seconds': round(gen_sec, 1),
        'total_seconds': round(time.time() - t0, 1),
        'peak_vram_mib_nvidia_smi': vram.peak_mib,
    }
    with open(os.path.splitext(args.output)[0] + '_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
