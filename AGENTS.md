# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Course team repo (MMDL). Assignment 1: build a reproducible evaluation pipeline for **Qwen3-VL-4B-Instruct on MMMU validation** (30 subjects × 30 = 900 questions). The same pipeline will later be re-run on fine-tuned checkpoints of this backbone, so prompt / decoding / parsing must stay fixed and fully documented — matching the official 67.4 is *not* the goal.

- Spec: `docs/assignment_guidance.md` (grading rubric, fixed constraints)
- Report template: `docs/SUBMISSION_TEMPLATE.md` → submit as `reports/mmmu_baseline.md`
- `assignment/assignment1.md` is an earlier copy of the template with team members filled in
- Deadline: 2026-09-28 23:59

## Commands

```bash
bash scripts/run_mmmu_eval.sh --model_path <HF id or ckpt dir> --data_root <HF datasets cache> --output_dir <dir> [--subjects Math] [--max_new_tokens N]   # GPU: infer once + score with both parsers
python scripts/score.py --pred <dir>/predictions.jsonl --parser qwen|mmmu --output_dir <dir>   # CPU re-score, no re-inference
python scripts/test_score.py   # CPU smoke check of prompt building + both scorers (needs numpy only)
```

`scripts/infer.py` (vLLM, GPU only) → `predictions.jsonl` + `predictions_meta.json`; `scripts/score.py` → `scores_{parser}.{csv,json,md}`. `scripts/third_party/` holds verbatim copies of the Qwen and MMMU parsers — never edit them; adapt in `score.py`. Current work/next steps: `HANDOFF.md`.

## Hard constraints (from the spec — do not relax)

- Model: `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, bf16, no quantization/casting. Never commit weights.
- Data: HF `MMMU/MMMU`, revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, split `validation`, each of the 30 subjects loaded as its own config. No filtering/subsampling.
- Sampling recipe must be Qwen's official one with a cited source (not ad hoc). `max_new_tokens` and `min_pixels`/`max_pixels` are free choices but need a documented infra trade-off.
- One-command reproduction; model path and data location passed as args/env vars, no hardcoded absolute paths.
- Overall score = macro average of the 30 subject accuracies (state the formula).

## Reference pipelines (sources to cite)

- **Sampling recipe (primary source)**: pinned model card `huggingface.co/Qwen/Qwen3-VL-4B-Instruct` @`ebb281ec` README "Generation Hyperparameters" has two recipes. **VL** (use for MMMU): temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5, out_seq_length 16384 — identical to its `generation_config.json`. **Text** (text-only benchmarks): 1.0 / 1.0 / 40 / 2.0, out_seq_length 32768; the Qwen3-VL tech report (arXiv 2511.21631 §5.11 Text-Centric Tasks) gives the same values for 8B/4B/2B instruct. No seed on the card; GitHub README says 3407, `evaluation/mmmu/run_mmmu.py` hardcodes vLLM `seed=42`.
- **Tech report Appendix B.1** lists the MMMU prompt: `<image>\nQuestion: {question}\nOptions:\n{options}\nPlease select the correct answer from the options above.` (images first). GitHub README: runtime vLLM, eval frameworks VLMEvalKit / lmms-eval.
- **Qwen official code**: `QwenLM/Qwen3-VL` → `evaluation/mmmu/` (vLLM). Prompt = VLMEvalKit `ImageMCQDataset.build_prompt` (`Question: ...\nOptions:\nA. ...\nPlease select the correct answer from the options above.`), all images placed before the text, `min_pixels=1280*28*28`, `max_pixels=5120*28*28`.
  - Parser is copied from VLMEvalKit `vlmeval/utils/matching_util.py` at ~commit `aca64dbb31` (2025-04), with one behavioral change: the `'A' in splits and len(splits) > 3` quantifier check runs unconditionally (upstream gates it on `VERBOSE`). Rule failure → GPT-3.5 judge (DashScope/OpenAI API) → unseeded random.
  - Open questions are scored via `MMMU_preproc`: rewritten as 2-choice (`A=gold`, `B="Other Answers"`) — lenient, and can false-positive when the response contains a standalone "A".
  - Loads VLMEvalKit's `MMMU_DEV_VAL.tsv` (dev+val, 1050 rows), not HF — must be replaced with the pinned HF dataset to satisfy the spec.
- **MMMU official**: `MMMU-Benchmark/MMMU` → `mmmu/utils/eval_utils.py` (`parse_multi_choice_response`, `parse_open_response`, `eval_open`). The prompt in `configs/llava1.5.yaml` is LLaVA-1.5's, and `run_llava.py` only feeds `image_1`, so use this repo for parsing only.

## Current state

- Branch `assignment1-vllm-eval`: vLLM pipeline in `scripts/` (Qwen official prompt + model-card VL recipe, pinned HF data, two parsers). Written on a laptop; vLLM path not yet run on GPU — see `HANDOFF.md`.
- Branch `origin/assignment1-eval` (`code/eval_mmmu.py`, `code/run.sh`, `results/mmmu_evaluation_report.csv`): HF `transformers.generate()` batch-1 baseline, overall 53.89 (485/900). Known issues: no dataset revision pin; the model card's *Text* recipe instead of the *VL* one (temp 1, top_p 1.0, top_k 40, presence 2.0; max_new_tokens 128); `presence_penalty` parsed but never passed to `generate`; no seed; prompt appends "Output ONLY the single option letter", which suppresses reasoning (likely the main cause of the gap); `min_pixels`/`max_pixels` placed in chat-template content are probably ignored by the HF processor; custom regex MC parser whose fallback matches any character; open answers scored by exact string match (list-valued gold answers like `"['Tampa', 'Florida']"` can never match). Keep its results as a "letter-only / short budget" control for gap analysis.

## Infra

Evaluation runs on a remote A100 SXM 80GB (this Mac has no CUDA). With vLLM, submit all 900 requests in a single `llm.generate()` call (per-question calls lose continuous batching); set per-request `SamplingParams(seed=...)`; log `finish_reason == "length"` counts for truncation analysis; save raw responses before parsing so re-scoring doesn't require re-inference.
