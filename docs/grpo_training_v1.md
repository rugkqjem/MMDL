# GRPO 학습 준비 현황 (v1)

> 2026-11-09 기준 · 브랜치 `rl-grpo` · 데이터 현황은 `docs/grpo_data_v1_status.md` 참고.  
> 한 줄 요약: **보상 함수와 학습 스크립트까지 준비 완료. 다음은 A100 한 대로 20스텝 파일럿.**  
> 2026-11-13 파일럿 완료: 결과는 **9절** 참고.

## 1. 무엇을 하려는가

- 목표: Qwen3-VL-4B-Instruct를 GRPO로 학습해서 **MMMU-val 점수(62.11)를 올리고, 반복 루프로 잘리는 응답을 줄이기**
- 학습 데이터: ViRL39K에서 고른 객관식 **1,052문제 (학습 780 + 검증 272)**. 8번 풀었을 때 맞는 답과 틀린 답이 섞여 나온 문제만 골랐어요
- 보상: 평가와 **같은 채점기**로 맞았는지 보고, 끝까지 쓰지 못하고 잘린 응답에는 감점

## 2. 결정한 것과 이유

| 항목 | 결정 | 이유 |
|---|---|---|
| 프레임워크 | **TRL `GRPOTrainer`** (EasyR1, verl 대신) | 보상 함수가 데이터셋의 모든 열(보기, 정답)과 토큰 id를 받아요. 평가 채점기를 고치지 않고 그대로 쓸 수 있고, 잘림도 EOS 토큰으로 정확히 판단할 수 있어요. EasyR1은 응답·토큰 수·정답 문자열만 넘겨줘요 |
| 실행 환경 | 평가 환경에 TRL/PEFT만 추가 | `-c requirements.lock.txt`로 torch·vLLM·transformers를 평가 버전 그대로 고정해요. TRL 1.17은 vLLM 0.22~0.33을 지원하고, 평가 환경은 0.30이에요 |
| 학습 샘플링 | **temperature 1.0, top_p 1.0, top_k 0, 패널티 없음** | 학습은 샘플링한 분포를 그대로 가정해요. presence_penalty를 걸면 루프를 억지로 막은 상태에서 학습하게 돼서, 모델이 스스로 루프를 피하는 법을 못 배워요. 평가는 원래 레시피 그대로 해요 |
| 잘림 처리 | 채점 **전에** 잘림부터 확인해서 감점(기본 −0.5). `mask_truncated_completions=False` | 잘린 응답을 학습에서 빼면(DAPO 방식) 루프를 고칠 신호가 사라져요 |
| LoRA 위치 | **LLM 디코더의 모든 linear** (q,k,v,o,gate,up,down), rank 16. ViT와 DeepStack merger는 동결 | "LoRA Without Regret": attention에만 걸면 rank를 올려도 성능이 떨어지고, RL은 rank 1~32면 충분해요. 루프는 이미지 인식보다 생성 습관 문제라서 vision 쪽을 건드릴 근거가 약해요 |
| 응답 길이 | 최대 8192토큰 (평가는 16384) | 롤아웃으로 문제를 고를 때 쓴 길이와 맞췄어요 |

## 3. 보상 함수 (`scripts/rl_reward.py`)

| 응답 | 보상 |
|---|---:|
| 잘림 (최대 길이에 닿았고 마지막 토큰이 EOS가 아님) | −0.5 |
| 끝까지 쓰고 정답 | 1.0 |
| 끝까지 쓰고 오답 / 파싱 실패 | 0.0 |

- 채점은 `score.py`의 `extract_final_answer` + `score_qwen`을 그대로 불러 써요(= 주 지표와 같은 방식)
- 주관식이 들어오면 오류를 내요. `score_qwen`은 주관식을 "정답 문자열이 들어 있으면 정답"으로 채점해서, RL에서는 보상 해킹이 되기 때문이에요

**검증 결과 (CPU)**
- MMMU baseline 객관식 847문제: 보상 함수 판정이 `scores_qwen.csv`와 **847/847 일치**
- ViRL 롤아웃(최종 학습셋 780문제 × 8 = 6,240개 응답): 롤아웃 때 판정을 **6,240/6,240 재현**
- 잘림 패널티 크기별로, 그룹 안 보상이 전부 같아서 학습 신호가 없는 문제 비율:

| 잘림 패널티 | 평균 보상 | 신호 없는 문제 |
|---:|---:|---:|
| 0 | 0.451 | 56 (7.2%) |
| 0.25 | 0.409 | 0 |
| 0.5 | 0.367 | 0 |
| 1.0 | 0.284 | 0 |

→ 패널티가 0보다 크기만 하면 모든 문제에 신호가 생겨요. 패널티 크기는 정답·오답·잘림이 섞인 그룹에서 잘림을 얼마나 세게 누를지만 정해요. 일단 0.5로 시작해요.
(이 롤아웃은 평가 샘플링 설정으로 뽑은 거라, 학습 샘플링에서의 실제 값은 파일럿에서 다시 봐요.)

## 4. 학습 설정 (`scripts/train_grpo.py` 기본값)

| 항목 | 값 |
|---|---|
| 그룹 크기 (문제당 응답 수) | 8 |
| 스텝당 응답 수 | 32 (= 4문제 × 8). 마이크로배치 2 × 누적 16 |
| vLLM 생성 단위 | 128개 응답(16문제)을 한 번에 생성해서 4스텝에 나눠 써요 |
| 학습률 | 1e-5, warmup 5스텝 후 고정 |
| 에폭 | 2 → 780 × 8 ÷ 32 = **195스텝/에폭, 총 390스텝** |
| loss | DAPO 방식(토큰 수로 정규화), KL 없음(beta 0), 그룹 단위 표준화 |
| vLLM | 학습 GPU에 같이 올림(colocate), 메모리 비율 0.35, 최적화 단계 동안 sleep |
| 체크포인트 | 25스텝마다 LoRA 어댑터 저장(최근 3개) |

**평가와 똑같이 맞춘 것**: 프롬프트(`--check_only`가 실제 프로세서로 평가 채팅 템플릿과 문자열을 비교), 이미지 해상도(평가와 같은 `qwen_vl_utils` 리사이즈를 미리 적용하고, 프로세서가 다시 줄이지 않는지 확인), 모델 revision

## 5. 파일과 실행 방법

| 파일 | 역할 |
|---|---|
| `scripts/rl_reward.py` | 보상 함수 |
| `scripts/test_rl_reward.py`, `scripts/check_rl_reward.py` | 보상 함수 테스트 / 실제 응답으로 검증 |
| `scripts/train_grpo.py` | GRPO 학습 (`--check_only`: GPU 없이 사전 검사) |
| `scripts/test_train_grpo.py` | 학습 스크립트 CPU 테스트 (리사이즈, LoRA 위치, 설정값) |
| `scripts/merge_lora.py` | LoRA를 원본 모델에 합쳐 평가용 체크포인트 만들기 |
| `server/rl_train_setup.sh` | 서버: 환경 → 다운로드 → CPU 검사 → **파일럿 20스텝** → 요약 |
| `requirements.rl.txt` | TRL, PEFT (평가 버전 고정 제약과 함께 설치) |

```bash
# 랩실 A100 (디스크 여유 60GB 이상). data/는 git에 없으니 먼저 올리기. WS = 작업 폴더(기본 /workspace)
scp -r ~/Desktop/MMMU/MMDL/data/rl <user>@<host>:<WS>/MMDL/data/
WS=<WS> nohup bash server/rl_train_setup.sh > rl_train_setup.log 2>&1 &

# 파일럿 결과를 보고 전체 학습 (같은 명령으로 다시 실행하면 이어서 학습)
python scripts/train_grpo.py --output_dir outputs/grpo/run1

# 체크포인트 평가 (기존 평가 스크립트 그대로)
python scripts/merge_lora.py --adapter outputs/grpo/run1/checkpoint-XXX --out outputs/grpo/run1/merged_XXX
bash scripts/run_mmmu_eval.sh --model_path outputs/grpo/run1/merged_XXX --model_revision '' \
    --data_root $HF_DATA --output_dir outputs/grpo_run1_XXX_mmmu_val
```

## 6. 파일럿에서 볼 것

셋업 스크립트가 마지막에 요약을 출력해요. 매 스텝 값은 `outputs/grpo/pilot_train/metrics.jsonl`에 있어요.

| 지표 | 의미 | 기대 |
|---|---|---|
| `frac_reward_zero_std` | 학습 신호가 없는 문제 비율 | 낮을수록 좋음. 평가 샘플링 기준 오프라인 값은 0% |
| `completions/clipped_ratio` | 잘림 비율 | 평가 샘플링 롤아웃은 16.8%. 패널티가 없어서 처음엔 더 높을 수 있고, 학습하면서 줄어야 해요 |
| `reward_parts/truncated` | 보상 함수가 판단한 잘림 비율 | `clipped_ratio`와 **같아야** 잘림 판단이 맞는 거예요 |
| `step_time` | 스텝당 시간 | × 390스텝 = 전체 학습 시간 |
| `max_mem_gb` | 메모리 최댓값 | 80GB 안에 들어와야 해요 |

메모리가 부족하면 `EXTRA="--per_device_train_batch_size 1 --gradient_accumulation_steps 32"`로 다시 돌려요(스텝당 32개 유지).

## 7. 아직 확인 안 된 것

- 실제 Qwen3-VL-4B + 실제 프로세서로는 아직 안 돌려봤어요. 프롬프트·이미지 일치(`--check_only`), 메모리, 스텝 시간은 **서버 파일럿에서 처음 확인**돼요
- 학습 샘플링(temperature 1.0)에서 잘림 비율이 얼마나 올라가는지
- 실제로 MMMU 점수가 오르는지 → 체크포인트 몇 개를 병합해서 평가해 봐야 해요

## 8. 다음 할 일

1. ~~랩실 A100에서 셋업 스크립트로 파일럿 20스텝~~ → 완료 (9절)
2. 파일럿 요약으로 전체 학습 시간 계산 → 팀에서 진행 여부 결정 (계산은 9.3절)
3. 전체 학습(390스텝) → 체크포인트 2~3개를 MMMU-val로 평가해 baseline 62.11과 비교
4. 최종 발표 슬라이드에 방법론(LoRA 위치, 샘플링, 보상 설계) 정리

## 9. 파일럿 결과 (2026-11-13)

- 환경: NVIDIA A100-SXM4-80GB 1장. `server/rl_train_setup.sh` 기본값 그대로(`SAMPLING=train`, `TRUNC_PENALTY=0.5`, 20스텝)
- 한 줄 요약: **20스텝 끝까지 오류 없이 돌았어요. 메모리 52.2GB, 스텝당 평균 527초. 보상은 오르고 잘림은 줄었지만, 잘림 판정 두 값이 서로 맞지 않아요.**

### 9.1 소요 시간

| 단계 | 시간 |
|---|---|
| 2~4. 환경 설치 · 다운로드 · CPU 검사 | 25분 (12 + 3.5 + 9.5) |
| 5. 학습 시작 준비 (1스텝 전까지) | 약 15분 |
| 5. 학습 20스텝 | 2시간 55분 31초 |

- CPU 검사 모두 통과: `check_rl_reward --mmmu` 847/847, `--check_only` 780문제 프롬프트·이미지 평가와 일치(가장 긴 프롬프트 5,330토큰)
- 학습 시작 준비 15분 중 약 9분은 `check_parity`예요. `--check_only`에서 한 검사를 학습 시작 때 한 번 더 해요
- 응답을 생성하는 스텝(1, 5, 9, 13, 17)은 약 15~19분, 나머지는 약 5~7분이에요

### 9.2 지표 (응답을 생성한 스텝)

| 스텝 | 평균 보상 | 정답 비율 | `frac_reward_zero_std` | `clipped_ratio` | `reward_parts/truncated` | 평균 길이 | 메모리 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.293 | 0.359 | 0.313 | 0.195 | 0.133 | 2,720 | 50.6GB |
| 5 | 0.355 | 0.453 | 0.063 | 0.195 | 0.195 | 4,067 | 50.8GB |
| 9 | 0.473 | 0.539 | 0.188 | 0.148 | 0.133 | 3,408 | 50.8GB |
| 13 | 0.512 | 0.547 | 0.125 | 0.078 | 0.070 | 3,504 | 52.2GB |
| 17 | 0.461 | 0.523 | 0.188 | 0.125 | 0.125 | 3,749 | 52.2GB |

정답 비율은 `reward_parts/correct`, 평균 길이는 `completions/mean_length`(토큰)예요.

- **잘림 판정 불일치**: `reward_parts/truncated`가 `clipped_ratio`와 같아야 하는데(6절), 5번 중 3번(1, 9, 13스텝) 달라요. 보상 함수가 항상 더 적게 잘림으로 판정해요. 확인 필요
- `frac_reward_zero_std`: 평균 0.175. 평가 샘플링 오프라인 값(0%)과 달리 학습 샘플링에서는 응답 8개의 보상이 모두 같은 문제가 생겨요
- 보상 0.293 → 0.461, 잘림 0.195 → 0.125로 좋은 방향이에요. 다만 스텝마다 다른 문제(20스텝에 약 80문제)라 난이도에 따른 흔들림이 섞여 있어요

### 9.3 전체 학습 시간 추정

셋업 요약의 `step_time 927s`는 응답을 생성한 스텝만의 평균이라, `195 × 927초`로 계산하면 두 배 가까이 과대 추정돼요. 실제 시간(20스텝 2시간 55분 31초, 스텝당 527초)으로 계산했어요.

| | 스텝 | 시간 | 비용 (A100 SXM 온디맨드 $2.19/시간) |
|---|---:|---:|---:|
| 1에폭 | 195 | 약 28.5시간 | 약 $62 |
| 2에폭 (현재 설정) | 390 | 약 57시간 | 약 $125 |

실행할 때마다 학습 시작 준비 약 15분이 더 들어요. 응답 길이에 따라 생성 시간이 크게 달라져서, 잘림이 줄면 짧아질 수 있어요.

## 참고

- TRL GRPO Trainer 문서: https://huggingface.co/docs/trl/en/grpo_trainer
- TRL "LoRA Without Regret" 가이드: https://huggingface.co/docs/trl/en/lora_without_regret (원문: Thinking Machines Lab, 2025)
- TRL vLLM 연동: https://huggingface.co/docs/trl/en/vllm_integration
