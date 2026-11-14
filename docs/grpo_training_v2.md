# GRPO 학습 준비 현황 (v2)

> 2026-11-14 기준 · 브랜치 `rl-grpo` · 이전 문서 `docs/grpo_training_v1.md` (설계·설정·1차 파일럿), 데이터 현황 `docs/grpo_data_v1_status.md`  
> 한 줄 요약: **1차 파일럿의 잘림 판정 불일치는 vLLM이 이미지 토큰을 두 번 펼친 버그 때문. 버그와 잘림 판정을 고쳤고, 재파일럿에서 IS 가중치가 낮아 `token_truncate`로 바꿔 20스텝 재파일럿.**

설계와 학습 설정은 v1의 2·4절과 같음. 이 문서는 문제, 원인, 수정, 재실행만 정리함.

## 1. 1차 파일럿에서 나온 문제

1차 파일럿(2026-11-13, A100 80GB, 20스텝)은 오류 없이 끝났지만 지표 두 개가 이상함.

| 문제 | 관찰 |
|---|---|
| 잘림 판정 불일치 | `reward_parts/truncated` < `clipped_ratio` (1, 9, 13스텝) |
| 학습 신호가 거의 0 | `importance_sampling_ratio` 평균 0.0~0.05 (정상이면 ≈ 1), grad_norm 1e-5 수준 |

분석 자료: 응답을 생성한 5스텝 × 128개 응답 원문과 매 스텝 지표 (`data/grpo_pilot_trunc_check.tgz`)

## 2. 분석 결과

### 2.1 잘림 판정 불일치: vLLM이 8192토큰 전에 응답을 끊음

| 지표 | 잘림 기준 |
|---|---|
| TRL `clipped_ratio` | 마지막 토큰이 EOS/pad가 아니면 잘림 (길이 무관) |
| 보상 `is_truncated` (v1) | 길이 ≥ 8192 **이고** 마지막 토큰이 EOS가 아니면 잘림 |

8192보다 짧은데 EOS 없이 끝난 응답이 차이 개수만큼 있었음.

| 스텝 | 차이 | 해당 응답 |
|---:|---:|---|
| 1 | 8 | 이미지 5장 문제 (`474c1aff…`): 응답 8개 모두 3,198토큰에서 끊김 |
| 9 | 2 | 이미지 4장 문제 (`d7679736…`): 5,403토큰에서 끊김 |
| 13 | 1 | 이미지 3장 문제 (`c8312d44…`) idx 56: 7,384토큰에서 끊김 |

- 같은 문제의 응답이 같은 길이에서 끊김 → 모델이 아니라 vLLM `max_model_len`(13,586 = 최장 프롬프트 5,330 + 8,192 + 64)에 걸린 것
- 모두 이미지가 여러 장인 문제

### 2.2 원인: vLLM이 이미지 토큰을 두 번 펼침

끊긴 길이로 거꾸로 계산하면 vLLM이 본 프롬프트는 10,388 / 8,183 / 6,202토큰임. 트레이너 프롬프트(최대 5,330)보다 훨씬 김.

1. TRL이 프로세서로 토큰화함 → 이미지가 이미 `<|image_pad|>` N개로 펼쳐져 있음
2. TRL이 펼쳐진 토큰 id와 원본 이미지를 **같이** vLLM에 넘김
3. vLLM이 `<|image_pad|>` 하나하나를 이미지 자리로 보고 **다시 펼침**

→ 프롬프트가 두 배 가까이 늘고 이미지 순서도 어긋남. vLLM은 꼬인 프롬프트로 응답을 만들었고, 트레이너는 정상 프롬프트로 확률을 계산함.

- 이미지 한 장 문제도 프롬프트는 꼬였음. 길이에 여유가 있어서 끊기지 않았을 뿐임
- 토큰당 확률 차이(`sampling_logp_difference`)가 평균 0.05, 최대 12.6으로 컸던 것도 이 영향으로 추정함

### 2.3 틀리게 매겨진 보상

보상은 잘림을 먼저 확인하고, 잘렸으면 파싱 결과와 상관없이 −0.5를 줌. 이 순서는 v1부터 맞았고, 잘림 판정이 틀렸던 것임. 1차 파일럿 응답 640개를 다시 채점한 결과는 아래와 같음.

| 스텝 | 응답 | v1 보상 | 수정 후 |
|---:|---|---:|---:|
| 1 | idx 32~39 (8개) | 0.0 | −0.5 |
| 9 | idx 50 (중간에 끊겼는데 D가 파싱돼 정답 처리) | **1.0** | −0.5 |
| 9 | idx 54 | 0.0 | −0.5 |
| 13 | idx 56 | 0.0 | −0.5 |

- v1 규칙으로 채점하면 파일럿 기록과 640/640 일치함. 수정 후 잘림 개수는 5스텝 모두 `clipped_ratio`와 같음 (25 / 25 / 19 / 10 / 16)
- 나머지 629개는 보상이 바뀌지 않음

### 2.4 학습 신호가 거의 0이었던 이유

TRL은 vLLM과 트레이너의 확률 차이를 "트레이너 확률 ÷ vLLM 확률" 가중치로 보정함. 기본값 `sequence_mask`는 이 가중치를 **응답 전체 단위**로 계산함.

- 토큰당 차이 0.05가 3,000~8,000토큰 동안 쌓이면 합이 ±3~10이 됨
- 가중치 = e^(합)이라 1에서 크게 멀어짐. 3보다 크면 그 응답을 버리고(0), 작으면 그대로 곱함 → 결국 0에 가까움

| 스텝 | 1 | 5 | 9 | 13 | 17 |
|---|---:|---:|---:|---:|---:|
| 가중치 평균 | 0.005 | 0.051 | 0.028 | 0.0 | 0.0002 |

→ v1 9.2절의 보상 상승(0.293 → 0.461)은 학습 효과가 아니라 스텝마다 다른 문제가 나온 데서 생긴 흔들림으로 봐야 함. 확률 차이의 원인이 2.2일 수 있어서 IS 모드는 일단 그대로 두고 재파일럿함 → 2.5절 결과를 보고 `token_truncate`로 변경.

### 2.5 재파일럿 1회차 (`sequence_mask`, 2026-11-14, 9스텝에서 중단)

| 스텝 | 잘림 (clipped = truncated) | 8192 전 끊김 | 확률 차이 평균 | IS 가중치 평균 | grad_norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.086 | 0 | 0.016 | 0.48 | 0.032 |
| 5 | 0.305 | 0 | 0.015 | 0.35 | 0.014 |
| 9 | 0.117 | 0 | 0.017 | 0.37 | 0.013 |

- 이미지 중복 버그는 해결됨: vLLM 프롬프트 검사 통과, 잘림 두 값 일치, 확률 차이 0.05 → 0.016, grad_norm 약 300배
- IS 가중치는 0.35~0.48에서 더 오르지 않음. 남은 확률 차이(토큰당 0.016)는 vLLM과 트레이너의 계산 차이로 보이며 학습해도 줄지 않음
- `sequence_mask`는 이 차이를 응답 전체에서 **합산**해 가중치가 3을 넘으면 응답을 통째로 버림 → 길수록 버려지기 쉬움. 잘림 감점을 받는 루프 응답(8192토큰)이 가장 많이 빠져서, 루프를 줄이는 신호가 가장 약해짐 (응답별 가중치가 로그에 없어서 추정)
- → `token_truncate`로 변경: 토큰마다 계산하고 3에서 자르기만 함. 길이와 상관없이 반영됨

## 3. 수정한 것

| 파일 | 수정 내용 |
|---|---|
| `scripts/rl_reward.py` | `is_truncated`: 길이 무관, **마지막 토큰이 EOS/pad가 아니면 잘림** (TRL과 같은 규칙). `trl_stop_ids()`로 TRL과 같은 종료 토큰 목록을 사용함 |
| `scripts/train_grpo.py` | vLLM에 넘기기 직전에 `<|image_pad|>` N개를 이미지당 1개로 줄임 → vLLM이 한 번만 펼침. 생성 후 vLLM 프롬프트와 트레이너 프롬프트가 토큰 단위로 같은지 확인하고, 다르면 에러를 냄. `--check_only`에 "pad 묶음 수 = 이미지 수" 검사를 추가함 |
| `scripts/train_grpo.py` (2.5절 후) | `--is_mode` 인자 추가, 기본값 `token_truncate` (TRL 기본값은 `sequence_mask`). `clip_max`는 TRL 기본값 3.0 |
| `scripts/test_rl_reward.py` | 새 잘림 규칙 테스트 |
| `scripts/test_vllm_prompt_fix.py` (신규) | 가짜 vLLM 엔진으로 버그 재현(38 → 56토큰)과 수정 후 복원을 확인함 |
| `server/rl_train_setup.sh` | 새 테스트 추가. `PILOT_NAME`으로 출력 폴더를 지정함(기존 파일럿이 있으면 중단). 요약에 스텝별 잘림 두 값, IS 가중치, 확률 차이를 추가함 |

## 4. 지켜볼 위험

| 위험 | 내용 | 볼 지표 / 대응 |
|---|---|---|
| 모델 책임이 아닌 잘림 | 1차의 11개는 vLLM 버그로 잘렸는데 감점됨 | 이번에 해결함. 이제 응답을 자르는 건 8192 예산뿐임 |
| 일찍 포기하고 찍기 | 0 > −0.5라서, 어려운 문제에서는 빨리 찍고 끝내는 쪽이 유리함 | `mean_length`는 급감하는데 `reward_parts/correct`가 그대로이거나, `parse_fail`이 늘면 패널티를 0.25로 낮춤 |
| 학습·평가 길이 예산 차이 | 평가는 16,384, 학습은 8,192. 평가에서 맞았을 8K~16K 응답도 −0.5를 받음 | 의도한 선택임. 발표 방법론에 명시할 것 |

## 5. 다시 실행하기

```bash
# 0) (로컬) 수정 코드를 rl-grpo에 커밋하고 push. 서버 스크립트가 rl-grpo를 pull함
git push origin rl-grpo

# 1) (서버) 재파일럿 20스텝. 새 출력 폴더를 씀 (1차 결과 outputs/grpo/pilot_train은 보존)
# 셋업 스크립트는 1단계에서 git pull로 자기 자신을 바꿔도 이미 연 이전 버전으로 끝까지 돎. 먼저 pull할 것
cd <WS>/MMDL && git pull --ff-only
WS=<WS> PILOT_NAME=pilot2_train nohup bash server/rl_train_setup.sh > rl_train_setup2.log 2>&1 &

# 2) 6절을 통과하면 전체 학습. 파일럿 체크포인트(checkpoint-20)를 run1로 옮기면 21스텝부터 이어서 학습함
#    (설정이 같고 lr이 상수라 이어가도 됨. transformers가 max_steps를 새 실행 값으로 다시 잡음)
mkdir -p outputs/grpo/run1 && cp -r outputs/grpo/pilot2_train/checkpoint-20 outputs/grpo/run1/
nohup python scripts/train_grpo.py --output_dir outputs/grpo/run1 > grpo_run1.log 2>&1 &

# 3) 체크포인트 평가
python scripts/merge_lora.py --adapter outputs/grpo/run1/checkpoint-XXX --out outputs/grpo/run1/merged_XXX
bash scripts/run_mmmu_eval.sh --model_path outputs/grpo/run1/merged_XXX --model_revision '' \
    --data_root $HF_DATA --output_dir outputs/grpo_run1_XXX_mmmu_val
```

- 환경·모델·이미지는 1차 때 받아둔 것을 재사용함. CPU 검사(4단계)를 통과해야 학습으로 넘어감
- 수정이 실제 vLLM에서 동작하지 않으면 첫 생성 스텝(약 15분)에서 `vLLM prompt N is X tokens, the trainer prompt Y` 에러로 멈춤
- `[vllm-fix] WARNING: ... not verified`가 뜨면 자동 비교를 못 한 것임. 6절의 잘림·확률 차이로 대신 확인함
- 메모리가 부족하면 `EXTRA="--per_device_train_batch_size 1 --gradient_accumulation_steps 32"`

## 6. 실험 후 확인해야 할 것

셋업 스크립트 마지막 요약과 `outputs/grpo/pilot2_train/metrics.jsonl`로 확인함.

| 확인 항목 | 1차 파일럿 | 통과 기준 | 통과 못 하면 |
|---|---|---|---|
| 프롬프트 수정 동작 | — | 첫 생성 스텝 에러 없음 | 에러 메시지의 토큰 수로 원인 조사 |
| `clipped_ratio` = `reward_parts/truncated` | 3/5 스텝 불일치 | 모든 스텝 일치 | 잘림 판정 재확인 |
| 8192 전에 같은 길이로 끊긴 응답 | 11개 | 0개 | 프롬프트 수정 재확인 |
| `sampling_logp_difference/mean` | 0.05 | 크게 감소 (≤ 0.01 기대) | 프롬프트 외 불일치 원인 조사 |
| `importance_sampling_ratio/mean` | 0.0~0.05 (재파일럿 1회차 0.35~0.48) | `token_truncate`에서는 토큰 평균이라 원래 ≈ 1. 판단 기준으로 쓰지 않음 | — |
| 잘림·길이 추세 | — | 스텝이 지날수록 `clipped_ratio`, `mean_length`가 줄어드는지 (같은 시드라 1차와 같은 스텝끼리 같은 문제) | 패널티·IS 설정 재검토 |
| `grad_norm` | 1e-5 수준 | 1차보다 확실히 큼 | IS 항목과 같이 봄 |
| `frac_reward_zero_std` | 0.175 | 1차 이하 | |
| `max_mem_gb` | 52.2GB | < 80GB | `EXTRA`로 배치 축소 |
| 20스텝 실제 시간 | 2시간 55분 | 1차와 비슷 | 전체 학습 시간 재계산 (요약의 `step_time`은 실제보다 짧게 나오니 실제 경과 시간을 씀) |
| 4절 위험 지표 | — | `mean_length` 급감이나 `parse_fail` 증가 없음 | 패널티 0.25 |


## 참고

- TRL 1.17.1: `trl/trainer/grpo_trainer.py` (`clipped_ratio`, importance sampling), `trl/generation/vllm_generation.py` (vLLM 입력 전달)
- TRL GRPO Trainer 문서: https://huggingface.co/docs/trl/en/grpo_trainer
