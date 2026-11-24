# RFT → DPO 실험 결과 (루프 줄이기)

> - 2026-11-18 · 브랜치 `rl-grpo` · 모델: Qwen3-VL-4B-Instruct · 평가: MMMU-val 900문제
> - 목표: 정답을 유지하면서, 반복하다 출력 한도에서 잘리는 응답(루프)을 줄이기. 잘림 기준은 평가 16,384토큰, 학습 롤아웃·재샘플링 8,192토큰
> - 모델: [yujin16/qwen3vl4b-mmmu-v1b](https://huggingface.co/yujin16/qwen3vl4b-mmmu-v1b) (v1b, LoRA 병합 완료)
> - 실행 방법: `docs/dpo_impl_notes.md` · 두 루프 쌍 방식 비교: `docs/dpo_loop_pairs_compare.md`

## 결론

- v1b(재샘플링 루프 쌍 + lr 5e-5)가 잘림을 90 → 35로 줄임 (RFT 대비 61% 감소). 같은 문제끼리 비교하면 66문제를 고치고 11문제가 새로 잘림(p < 0.001).
- 정확도는 64.89%. 베이스라인 대비로는 의미 있게 상승(정답 +73/−48, p = 0.03). 하지만 DPO 직전 모델인 RFT 대비로는 +62/−46(p = 0.15)이라, DPO 단계만의 정확도 효과는 아직 확정은 아님.
- 부작용은 없었음. 원래 끝까지 답하던 문제의 정확도와 응답 길이가 그대로.
- cut 방식 루프 쌍(v0)과 lr 1e-5(v0, v1)로는 효과가 없었음.

## 결과

Qwen 파서 + 최종 답 추출(`scores_qwen`) 기준. 평가는 모델당 1회.

| 모델 | 학습 | 정확도 | 맞힌 문제 | 잘림 |
|---|---|---|---|---|
| 베이스라인 | – | 62.11 | 559 | 85 |
| RFT | 베이스라인 + RFT | 63.11 | 568 | 90 |
| v0 | RFT + DPO (cut 방식 루프 쌍), lr 1e-5 | 62.00 | 558 | 95 |
| v1 | RFT + DPO (재샘플링 방식 루프 쌍), lr 1e-5 | 63.11 | 568 | 78 |
| **v1b** | **v1과 같은 데이터, lr 5e-5** | **64.89** | **584** | **35** |

### RFT 대비 같은 문제끼리 비교 (부호 검정)

| 모델 | 정답 +/− | p | 잘림 고침/새로 생김 | p |
|---|---|---|---|---|
| v0 | +52 / −62 | 0.40 | 38 / 43 | 0.66 |
| v1 | +44 / −44 | 1.00 | 35 / 23 | 0.15 |
| **v1b** | **+62 / −46** | 0.15 | **66 / 11** | **< 0.001** |

베이스라인 대비 정답 +/−: RFT +66/−57 (p = 0.47), v1 +64/−55 (p = 0.46), **v1b +73/−48 (p = 0.03)**

### v1b 부작용 확인 (RFT와 비교)

| 확인 | RFT | v1b | 해석 |
|---|---|---|---|
| 둘 다 끝까지 답한 799문제 정확도 | 68.6 | 68.8 | 일찍 찍고 끝내는 부작용 없음 |
| 응답 길이 중앙값 (토큰) | 433.5 | 429.5 | 전체적으로 짧아진 게 아님. 루프 문제만 고쳐짐 |
| 고친 루프 66문제 중 정답 | – | 28 (42%) | 정확도 상승분이 여기서 나옴 |

## 무엇을 했나

### 1. RFT (rejection sampling fine-tuning)

- 데이터: 최종 학습셋(ViRL39K 780문제) 롤아웃 중 **끝까지 쓴 정답 응답, 문제당 최대 2개** → 1,337개 / 724문제 (`scripts/make_rft_data.py`). 새로 생성한 건 없음.
- 학습: `scripts/train_ft.py`, LoRA lr 1e-4, 1에폭, 배치 1 × 누적 8 (168스텝). loss 약 0.26에서 시작.
- 결과: 정확도와 잘림 모두 베이스라인과 차이 없음. DPO의 출발점 역할.

### 2. DPO 쌍

모든 버전에 **정답 쌍 531개**가 같이 들어감. 같은 문제에서 끝까지 쓴 정답 응답(chosen)과 오답 응답(rejected)이고, 길이 비율은 1.5배 이내.

| | v0: cut | v1·v1b: 재샘플링 |
|---|---|---|
| 루프 쌍 | 230개 / 160문제 | 246개 / 246문제 |
| chosen | 같은 문제의 다른 정답 응답 | 루프 응답의 갈림길에서 다시 생성해 끝까지 쓴 이어쓰기 |
| rejected | 루프 응답 처음부터, 첫 "보기에 없음" + 1,500토큰까지 | 같은 갈림길 이후의 원래 루프 응답 |
| 공통 앞부분 | 없음 | 갈림길까지 (손실 계산에서 빠짐) |

재샘플링 (`scripts/dpo_resample.py`)
- 모델: 원본 Qwen3-VL-4B-Instruct (RFT와 병행하려고). 평가와 같은 샘플링 설정.
- 잘린 응답 문제당 1개(375개). 응답의 60% 지점부터 4번 이어쓰기 → 하나도 못 빠져나오면 45% 지점.
- 375개 중 253개가 빠져나옴. 빈 응답이거나 답이 없는 이어쓰기를 빼고 루프 쌍 246개. chosen 중 정답 157개(64%).

### 3. DPO 학습

- `scripts/train_dpo.py` (TRL 1.17.1 DPOTrainer). 기준 모델 = RFT 모델(LoRA 어댑터를 끈 상태).
- sigmoid DPO, beta 0.1, SFT 항 없음, cosine, warmup 10%, 1에폭, 배치 1 × 누적 16 (48~49스텝). LoRA r16 / alpha 32, LLM 디코더 전체 linear, vision 동결.
- v1b만 lr 5e-5, 나머지는 1e-5.
- 학습 로그 차이: lr 1e-5에서는 `rewards/margins`가 0 ~ 0.1, 5e-5에서는 0.4 ~ 1.15로 커짐. 기울기 크기(grad_norm 12~20)가 1.0으로 잘리기 때문에, 1e-5로는 모델이 거의 안 움직였음.

## 한계와 주의점

- **평가 1회**
- **공정하지 않은 비교**: v0에 lr 5e-5를 적용한 버전은 돌리지 않았음. "재샘플링 방식이 더 낫다"는 lr 1e-5 조건에서만 말할 수 있음.
- **재샘플링 모델**: 재샘플링은 RFT 모델이 아니라 원본 모델로 함. 두 모델의 차이가 작아서(RFT 정확도·잘림이 베이스라인과 같음) 영향은 작을듯.

## 재현

```bash
# 데이터 (CPU): RFT 데이터, v0 쌍
python scripts/make_rft_data.py
python scripts/dpo_pairs.py --loop_mode cut --out outputs/dpo/pairs_cut.jsonl
# RFT
python scripts/train_ft.py --train_data data/rft/rft_train.jsonl --image_root data/virl39k_raw \
  --output_dir outputs/rft/merged --lora_llm --gradient_checkpointing --max_length 12288 --save_steps 50
# 재샘플링 → v1b
python scripts/dpo_resample.py --model_path Qwen/Qwen3-VL-4B-Instruct --per_question 1 --k 4 --max_branches 2
python scripts/dpo_pairs.py --resample outputs/dpo/resample.jsonl --out outputs/dpo/pairs_resample.jsonl
python scripts/train_dpo.py --model_path outputs/rft/merged --pairs outputs/dpo/pairs_resample.jsonl \
  --output_dir outputs/dpo/v1b --learning_rate 5e-5
python scripts/merge_lora.py --model_path outputs/rft/merged --adapter outputs/dpo/v1b/final_adapter --out outputs/dpo/v1b/merged
# 평가
bash scripts/run_mmmu_eval.sh --model_path outputs/dpo/v1b/merged --model_revision '' --data_root <HF 캐시> --output_dir outputs/dpo_v1b_mmmu_val
```

- 서버 묶음 `server_bundle_rft_dpo_20261117.tar.gz`와 분석 묶음 `analysis_bundle_20261118.tar.gz`(평가 결과, 학습 로그, 학습 쌍)는 팀에 따로 공유.
- A100 80GB 1장 기준: RFT 약 30분, 재샘플링 약 1시간 40분, DPO 학습 약 1시간 40분, 평가 30분~1시간.
