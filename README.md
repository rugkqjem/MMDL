# MMDL: Qwen3-VL-4B의 반복 루프 줄이기 (RFT → DPO)

멀티모달 딥러닝 수업 1조 팀 레포입니다. **Qwen3-VL-4B-Instruct**가 MMMU에서 같은 확인을 되풀이하다 출력 한도에서 잘리는 문제(반복 루프)를,
모델 자신의 응답으로 만든 학습 데이터(RFT → DPO)로 줄였습니다. 평가는 과제 1에서 만든 MMMU 평가 파이프라인을 그대로 씁니다.

- **팀원**: 김유진, 류승희, 송지훈
- **최종 모델**: [yujin16/qwen3vl4b-mmmu-v1b](https://huggingface.co/yujin16/qwen3vl4b-mmmu-v1b) (LoRA 병합 완료, 바로 평가 가능)
- **학습 데이터 · 중간 산출물**: [yujin16/mmdl-data](https://huggingface.co/datasets/yujin16/mmdl-data) (git에 넣기 큰 파일)

## 결과

채점은 모두 **최종 답 추출 → Qwen 공식 규칙 파서** (`scores_qwen.json`), 점수는 전체 정답 비율(micro)입니다.
MMMU val은 과목당 30문제라 과목 평균(macro)과 같습니다. 평가는 모델당 1회입니다.

| 모델 | MMMU val (900) | MMMU-Pro 보기 10개 (1,730) | MMMU-Pro vision (1,730) | MMMU test (10,500) |
|---|---|---|---|---|
| 베이스라인 (Qwen3-VL-4B-Instruct) | 62.11 | 50.98 | 평가 중 | 평가 중 |
| **최종 모델 (v1b)** | **64.89** | **54.51** | 평가 중 | 평가 중 |

- **잘린 응답** (출력 한도 16,384토큰): MMMU val 85 → 35, MMMU-Pro 보기 10개 263 → 129
- **같은 문제끼리 비교 (부호 검정)**: MMMU val 정답 +73 / −48 (p = 0.03), MMMU-Pro 정답 +173 / −112 (p < 0.001)
- **MMMU-Pro 중 1,153문제** (MMMU test 출신, 설정을 고를 때 안 본 문제): 52.65 → 56.46 (p = 0.003).
  나머지 577문제는 MMMU val 출신이라 따로 봤습니다
- **부작용 없음**: 두 모델이 모두 끝까지 답한 문제의 정확도는 그대로 (MMMU val 68.6 → 68.8 (RFT 대비), MMMU-Pro 59.3 → 59.2)
- 평가 방식: MMMU val로 학습률 · 쌍 방식을 고르고, MMMU-Pro와 MMMU test로 최종 평가

### MMMU val 전체 실험

| 모델 | 학습 | 정확도 | 잘림 |
|---|---|---|---|
| 베이스라인 | – | 62.11 | 85 |
| GRPO 100스텝 | 온라인 RL, 잘림 감점 | 62.89 | 87 |
| RFT | 끝까지 쓴 자기 정답 응답으로 SFT | 63.11 | 90 |
| v0 | RFT + DPO (cut 방식 루프 쌍), lr 1e-5 | 62.00 | 95 |
| v1 | RFT + DPO (재샘플링 루프 쌍), lr 1e-5 | 63.11 | 78 |
| **v1b** | **RFT + DPO (재샘플링 루프 쌍), lr 5e-5** | **64.89** | **35** |

자세한 수치와 통계: [docs/rft_dpo_results.md](docs/rft_dpo_results.md) · GRPO: [docs/grpo_run1_100_result.md](docs/grpo_run1_100_result.md)

## 방법

```
ViRL39K (38,870) → 객관식 · MMMU 중복 제거 · 보기 3개 이상 (8,982) → 기본 모델로 롤아웃 (3,272)
  → 정답 · 오답 · 잘림이 섞인 문제, 채점 형식 · 라벨 오류 제외 → 학습 780 + 검증 272
      ├─ GRPO   보상 정답 1 / 오답 0 / 잘림 −0.5 (100스텝, 잘림 줄지 않아 중단)
      └─ RFT    롤아웃 중 끝까지 쓴 정답 응답, 문제당 최대 2개 (1,337개)로 SFT
           └─ DPO  정답 쌍 531 + 루프 쌍 246
                   루프 쌍: 잘린 응답의 60% 지점(안 되면 45%)에서 4번 이어쓰기 →
                            끝까지 쓴 이어쓰기 = chosen, 원래 루프 = rejected (앞부분은 손실에서 제외)
```

- LoRA r16 / alpha 32, LLM 디코더의 모든 linear. 비전 인코더 · merger는 동결
- DPO: sigmoid, beta 0.1, 기준 모델 = RFT 모델, 1에폭, 배치 1 × 누적 16
- 학습 데이터는 MMMU dev · val · test와 MMMU-Pro를 기준으로 중복 제거 (13-gram, MinHash, pHash, CLIP)

## 재현

모든 단계는 A100 80GB 1장 기준입니다. 학습 · 데이터 스크립트는 평가 코드(`infer.py`, `score.py`)를 import해서 프롬프트 · 이미지 처리 · 채점을 평가와 똑같이 씁니다.

### 0. 환경

```bash
uv venv --seed --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.lock.txt                           # 평가 환경 (vLLM 0.30, transformers 5.17, torch 2.13, CUDA 13)
uv pip install -r requirements.rl.txt -c requirements.lock.txt   # 학습 (TRL 1.17.1, PEFT)
```

### 1. 평가

```bash
# MMMU val (보고 점수) / MMMU test
bash scripts/run_mmmu_eval.sh --model_path Qwen/Qwen3-VL-4B-Instruct --data_root <HF 캐시> --output_dir outputs/baseline_mmmu_val
bash scripts/run_mmmu_eval.sh --model_path Qwen/Qwen3-VL-4B-Instruct --data_root <HF 캐시> \
  --split test --dataset_revision main --output_dir outputs/baseline_mmmu_test
# MMMU-Pro (test만 있음): standard = 보기 10개, vision = 문제 전체가 이미지 한 장
bash scripts/run_mmmu_eval.sh --model_path Qwen/Qwen3-VL-4B-Instruct --data_root <HF 캐시> --pro standard --output_dir outputs/baseline_pro_standard
bash scripts/run_mmmu_eval.sh --model_path Qwen/Qwen3-VL-4B-Instruct --data_root <HF 캐시> --pro vision --output_dir outputs/baseline_pro_vision
```

- 최종 모델은 `--model_path yujin16/qwen3vl4b-mmmu-v1b --model_revision ''`
- MMMU test 정답은 2026-02-12에 HF에 공개돼서 `--dataset_revision main`이 필요합니다 (고정 revision에는 정답이 없음)
- MMMU-Pro 프롬프트는 공식 CoT 지시문 (`scripts/infer_mmmu_pro.py`)
- 점수: `outputs/<이름>/scores_qwen.json`의 `micro_acc`. 재채점은 `python scripts/score.py --pred <predictions.jsonl> --parser qwen --output_dir <폴더>`

### 2. 학습 데이터

롤아웃(3,272문제 × 최대 8번)은 A100으로 5시간 넘게 걸려서, 결과를 [yujin16/mmdl-data](https://huggingface.co/datasets/yujin16/mmdl-data)에 올려 두었습니다.
레포 루트에서 받으면 아래 단계가 쓰는 경로에 그대로 들어갑니다.

```bash
hf download yujin16/mmdl-data --repo-type dataset --local-dir .
hf download TIGER-Lab/ViRL39K --repo-type dataset --local-dir data/virl39k_raw    # 이미지
```

처음부터 만들려면: `server/rl_data_setup.sh` (변환 → 중복 제거 → 파일럿) → `scripts/rl_rollout_filter.py` (롤아웃) → `scripts/rl_select.py --keep_trunc_mixed --val_size 300`
→ `data/rl/exclude_format_only.txt`, `exclude_label_issues.txt`의 문제 제외. 자세한 과정: [docs/grpo_data_v1_status.md](docs/grpo_data_v1_status.md), [scripts/RL_DATA_README.md](scripts/RL_DATA_README.md)

### 3. RFT

```bash
python scripts/make_rft_data.py                                   # → data/rft/rft_train.jsonl (1,337개, CPU)
python scripts/train_ft.py --train_data data/rft/rft_train.jsonl --image_root data/virl39k_raw \
  --output_dir outputs/rft/merged --lora_llm --gradient_checkpointing --max_length 12288 --save_steps 50
```

### 4. DPO (v1b)

```bash
python scripts/loop_onset.py                                      # 잘린 응답의 갈림길 후보 (CPU)
python scripts/dpo_resample.py --model_path Qwen/Qwen3-VL-4B-Instruct --per_question 1 --k 4 --max_branches 2
python scripts/dpo_pairs.py --resample outputs/dpo/resample.jsonl --out outputs/dpo/pairs_resample.jsonl
python scripts/train_dpo.py --model_path outputs/rft/merged --pairs outputs/dpo/pairs_resample.jsonl \
  --output_dir outputs/dpo/v1b --learning_rate 5e-5
python scripts/merge_lora.py --model_path outputs/rft/merged --adapter outputs/dpo/v1b/final_adapter --out outputs/dpo/v1b/merged
bash scripts/run_mmmu_eval.sh --model_path outputs/dpo/v1b/merged --model_revision '' --output_dir outputs/v1b_mmmu_val
```

- v0 (cut 방식): `python scripts/dpo_pairs.py --loop_mode cut --out outputs/dpo/pairs_cut.jsonl` 후 같은 `train_dpo.py` (lr 1e-5)
- 단계별 설명: [docs/dpo_impl_notes.md](docs/dpo_impl_notes.md) · 두 쌍 방식 비교: [docs/dpo_loop_pairs_compare.md](docs/dpo_loop_pairs_compare.md)
- 걸린 시간: RFT 약 30분, 재샘플링 약 1시간 40분, DPO 약 1시간 40분, MMMU val 평가 30분~1시간

### 5. GRPO (참고: 효과 없음)

```bash
python scripts/train_grpo.py --output_dir outputs/grpo/run1      # 서버 셋업 · 파일럿: server/rl_train_setup.sh
```

설계 · 문제 · 결과: [docs/grpo_training_v1.md](docs/grpo_training_v1.md), [docs/grpo_training_v2.md](docs/grpo_training_v2.md), [docs/grpo_run1_100_result.md](docs/grpo_run1_100_result.md)

## 고정 설정

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3-VL-4B-Instruct` @`ebb281ec70b05090aa6165b016eac8ec08e71b17`, bf16 |
| 데이터 | HF `MMMU/MMMU` @`98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` (val), `main` (test) · `MMMU/MMMU_Pro` test |
| 프롬프트 | MMMU: Qwen 공식 `run_mmmu.py`의 `build_mmmu_prompt()` · MMMU-Pro: 공식 `prompts.yaml`의 CoT. 이미지는 텍스트 앞 |
| 샘플링 | temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5, seed 3407 |
| 생성 예산 | 평가 16,384토큰 (`max_model_len` 32768) · 학습 롤아웃 · 재샘플링 · GRPO 8,192토큰 |
| 이미지 | `min_pixels` 1280·28·28, `max_pixels` 5120·28·28 |
| 채점 | 최종 답 추출 → Qwen 규칙 파서. judge · 무작위 선택 없이 실패는 오답 |
| GPU | A100-SXM4-80GB 1장 |

## 폴더 구조

```
scripts/
  run_mmmu_eval.sh       추론 + 채점 한 번에 (--split test, --pro standard|vision)
  infer.py · infer_mmmu_pro.py · score.py      평가 (과제 1)
  rl_prepare_virl39k.py · decontaminate.py · rl_rollout_filter.py · rl_select.py   학습 데이터
  rl_reward.py · train_grpo.py                 GRPO
  make_rft_data.py · train_ft.py               RFT
  loop_onset.py · dpo_resample.py · dpo_pairs.py · train_dpo.py · merge_lora.py   DPO
  test_*.py              CPU 테스트
server/                  서버 셋업 스크립트
docs/                    단계별 설계 · 결과 문서
reports/                 과제 1 보고서 (베이스라인 재현, 오답 분석)
outputs/                 과제 1 평가 결과 (베이스라인 1차 · 재실행 · 32k)
```

## 과제 1: 베이스라인 재현

공식 보고 67.4 대비 우리 재현 62.11 (MMMU val). 환경 · 설정 · 격차 분석은 [reports/mmmu_baseline.md](reports/mmmu_baseline.md),
오답 341개 원인 분류는 [reports/mmmu_eval_analysis.md](reports/mmmu_eval_analysis.md)에 있습니다.
