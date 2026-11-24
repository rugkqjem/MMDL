"""Export an HF MMMU split to the qwen-vl-finetune JSONL that train_ft.py reads (images saved as PNG).

The user turn is built by infer.build_messages(), so training prompts match the evaluation prompts exactly. The target
is `Answer: <gold>`, a marker score.py's extract_final_answer() recognizes. Never train on `validation`: that is the
split run_mmmu_eval.sh reports. `dev` (5 per subject, 150 total) is the only other split with answers, so it is for
pipeline checks; real training data comes from elsewhere in the same format.

    python scripts/prepare_mmmu_sft.py --split dev --output_dir data/mmmu_dev_sft
    # -> data/mmmu_dev_sft/train.jsonl + data/mmmu_dev_sft/images/*.png  (train_ft.py --image_root data/mmmu_dev_sft/images)
"""
import argparse
import json
import os

from infer import DATASET_REVISION, SUBJECTS, build_messages


def to_row(sample, image_dir):
    messages, prompt, _ = build_messages(sample, None, None)
    images = []
    for i, item in enumerate(c for c in messages[0]['content'] if c['type'] == 'image'):
        name = f"{sample['id']}_{i + 1}.png"
        item['image'].convert('RGB').save(os.path.join(image_dir, name))
        images.append(name)
    return {'id': sample['id'], 'image': images,
            'conversations': [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': f"Answer: {sample['answer']}"}]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='dev', choices=['dev'], help='validation is the eval split; test has no answers')
    p.add_argument('--data_root', default=None, help='HF datasets cache dir for MMMU (default: HF cache)')
    p.add_argument('--dataset_revision', default=DATASET_REVISION)
    p.add_argument('--subjects', nargs='+', default=SUBJECTS)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    from datasets import load_dataset
    image_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    n = 0
    with open(os.path.join(args.output_dir, 'train.jsonl'), 'w') as f:
        for subject in args.subjects:
            ds = load_dataset('MMMU/MMMU', subject, split=args.split, revision=args.dataset_revision,
                              cache_dir=args.data_root)
            for sample in ds:
                f.write(json.dumps(to_row(sample, image_dir), ensure_ascii=False) + '\n')
                n += 1
    print(f'{n} samples -> {args.output_dir}/train.jsonl')


if __name__ == '__main__':
    main()
