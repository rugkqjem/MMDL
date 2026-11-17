# DPO 파이프라인 구현 명세 (코드 기준)

> 2026-11-16 · 브랜치 `rl-grpo` · 이 문서는 **코드가 실제로 하는 일**만 적어요. 주의점 조사 문서와 항목별로 대조하는 용도예요.
> 숫자는 맥에서 롤아웃 파일로 돌린 결과예요(CPU 단계만). GPU 단계는 아직 안 돌렸어요.

## 전체 흐름

```
롤아웃 응답 (원본 모델, ViRL, 평가 샘플링, 8192토큰)
  ├─ [1] loop_onset.py      잘린 응답에서 루프 판정 + 반복 시작 위치 + 갈림길 후보        CPU
  ├─ [2] dpo_resample.py    갈림길에서 이어쓰기 k번 → 빠져나오는 지점 찾기              GPU (RFT 모델로)
  ├─ [3] dpo_pairs.py       정답 쌍 + 루프 쌍 만들기, 규칙 검사                        CPU
  └─ [4] train_dpo.py       TRL DPOTrainer + LoRA                                    GPU
검증: test_dpo.py (CPU), train_dpo.py --check_only (실제 프로세서)
```

문제 범위는 모든 단계에서 **최종 학습셋 `data/rl/train_final.jsonl`(780문제)**만 써요. 검증셋 272문제는 쓰지 않아요.

---

## [1] 루프 신호와 갈림길 후보 (`scripts/loop_onset.py`)

- **입력 / 출력**: `rollouts_resp.jsonl` + `pilot_resp.jsonl`(두 파일에 겹치는 문제 없음 확인) → `outputs/dpo/loop_candidates.jsonl` (잘린 응답마다 한 줄)
- **대상**: `finish_reason == 'length'`인 응답(EOS 없이 8192토큰 도달) → 1,044개. **전부 다음 단계로 넘김** (루프 판정으로 거르지 않음)
- **정답지 결과 (2026-11-17, 사람 라벨 17개)** → 설계를 바꾼 이유
  - 라벨한 잘린 응답 **17개 전부 루프**(A1 1, A2 6, B 3, C 4, D 1, E 2). G(계산하다 한도 도달)는 아직 0개
  - 규칙에 안 걸린 그룹에서 뽑은 5개도 전부 루프 → 기존 규칙이 루프를 놓치고 있었음
  - 루프 잡은 수: 기존 규칙(`classify_errors`) 12, 반복 규칙 14. 놓친 건 모두 **A2**(가설을 말만 바꿔 다시 세움)
  - 반복 시작 위치는 텍스트 규칙으로 잘 안 맞음: 반복 규칙은 사람과 300자 이내 0/14, 최대 ±17,000자 차이. 밀도 규칙(아래)도 2,000자 이내가 9/16
  - 결론: 사람이 보는 루프 시작은 "Wait — perhaps…"처럼 **글자가 반복되기 전부터** 시작돼서 텍스트로 정확히 못 찾음 → **위치를 찾지 않고 재샘플링으로 결정**
- **기록하는 신호** (정보용, 거르는 데는 안 씀)
  - 단위: 줄 → 400자 넘는 줄은 문장 끝에서 → 그래도 길면 200자 조각
  - 반복 단위: 정규화한 텍스트가 앞에 나왔음, 또는 단어 5-gram의 60% 이상이 앞에 나왔음
  - 재시작 단위: "Wait", "Perhaps", "Alternatively", "But maybe", "Let me re…" 같은 망설임·재시작 표현으로 시작
  - `onset_char`(반복 규칙): 반복 단위이면서 그 뒤 남은 글자의 70% 이상이 반복인 첫 위치 → 521개(50%)에서 찾음
  - `dense_char`(밀도 규칙): 2,000자 창에서 반복+재시작이 70% 이상이고, 그 뒤로도 평균 60% 이상 유지되는 첫 위치 → 658개에서 찾음. 사람 라벨과 중앙값 +247자
  - `old_rule_loop`: 기존 규칙 결과 (346개)
- **갈림길 후보**: 응답 길이의 **60 / 45 / 30 / 15 %** 지점의 줄 시작. 늦은 것부터. 정답지 17개 기준으로 응답당 0~4개가 사람이 표시한 루프 시작보다 뒤(루프 안)에 있음
- `--compare_gold`: 사람 라벨과 비교 (기존·반복 규칙의 잡음/놓침, 두 규칙의 위치 오차, 유형별 결과, 루프 안에 들어간 후보 수)

## [2] 갈림길 재샘플링 (`scripts/dpo_resample.py`, GPU)

- **입력 / 출력**: `loop_candidates.jsonl` → `outputs/dpo/resample.jsonl` (+ `_meta.json`에 샘플링 설정 기록)
- **모델**: `--model_path` 필수. **DPO를 시작할 모델(RFT 병합 체크포인트)**로 돌리는 게 원칙
- **대상 선택**: 문제당 잘린 응답 최대 2개(텍스트 규칙에 걸린 것 우선), 응답당 갈림길 후보 최대 4개 → **615개 응답, 375문제**. `--only_flagged`면 규칙에 걸린 것만(372개)
- **프롬프트**: `infer.prepare_sample`(평가와 같은 텍스트, 이미지 리사이즈) + 원래 응답의 갈림길까지 앞부분을 assistant 답의 일부로 이어 붙임
- **샘플링**: 평가 설정 그대로(temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5). 최대 토큰 = 8192 − 앞부분 토큰 수(롤아웃과 같은 총 예산). 남은 예산이 512토큰 미만인 후보는 건너뜀
- **탈출** = 이어쓰기가 EOS로 끝남. 정답 여부는 앞부분+이어쓰기 전체를 평가 채점기로 채점해서 기록
- **후보 시도 순서**: 늦은 후보(60%)부터. 하나라도 탈출하면 그 후보를 채택하고 멈춤. 전부 다시 잘리면 한 단계 앞 후보로 → **"실제로 빠져나올 수 있는 가장 늦은 지점"이 채택됨**. 루프 시작 위치를 텍스트로 찾는 대신 이 행동 기준을 씀
- **비용 추정**: 응답 615개 × 8개 × 최대 4라운드. 늦은 후보일수록 남은 예산이 작아 이어쓰기가 짧음 (아직 안 재봄)
- **확인 못 한 것**: 실제 탈출률, 소요 시간, 앞부분을 붙였을 때 토큰 경계가 원래와 같은지(줄 끝에서 자르므로 문제없을 것으로 봄)

## [3] DPO 쌍 만들기 (`scripts/dpo_pairs.py`, CPU)

- **입력 / 출력**: `train_final.jsonl`, 롤아웃 응답, (있으면) `resample.jsonl` → `outputs/dpo/pairs.jsonl` + `pairs_report.md`
- **정답 쌍** (문제당 최대 1개)
  - chosen = EOS로 끝나고 평가 채점기로 정답
  - rejected = EOS로 끝나고, 답을 **규칙으로 파싱**했고(랜덤 대체 아님 → 파싱 실패 제외), 오답
  - 두 응답 토큰 길이 비율 1.5배 이내, 그중 길이가 가장 비슷한 쌍. 가장 짧은 정답을 고르지 않음
  - 잘렸는데 채점상 정답인 응답은 chosen이 될 수 없음 (EOS 조건)
  - **결과: 531쌍(531문제)**, 길이 비율 중앙값 1.06, chosen 1,643 / rejected 1,606토큰(중앙값)
- **루프 쌍** (문제당 최대 2개)
  - 앞부분(prefix) = 원래 루프 응답의 채택된 갈림길까지. **프롬프트 쪽에 들어가서 손실에서 상쇄됨**
  - chosen = 그 갈림길에서 탈출한 이어쓰기 중 하나(정답 우선, `--loop_chosen_correct`면 정답만)
  - rejected = 같은 갈림길에서 원래 응답이 이어간 부분. **EOS 없음**. chosen 이상, 최소 2,000자. 그 구간의 반복+재시작 비율이 50% 이상이 될 때까지 1,000자씩 늘림(원래 응답 끝까지 가면 거기서 멈춤). 단어 중간에서 안 자름
  - 결과: 아직 없음(재샘플링 전). 가짜 재샘플링 80개로 확인: rejected 중앙값 약 3,000자, 23개는 끝까지 늘려도 50% 미만(글자 반복이 적은 A2형으로 보임) → 끝까지 쓴 원래 응답 그대로 rejected
- **루프 쌍, cut 방식** (`--loop_mode cut`, 승희 방식, "SFT → DPO 계획 초안"의 쌍 A. 비교는 `docs/dpo_loop_pairs_compare.md`)
  - chosen = 같은 문제의 다른 정답 응답(EOS로 끝남). "보기 불일치 후 회복" 응답 우선, 쌍마다 다른 chosen
  - rejected = 보기 불일치 표현(`MISMATCH` 정규식)이 있는 잘린 응답의 처음부터, max(chosen 토큰, 첫 불일치 + 1,500토큰), 최대 4,095토큰, 줄바꿈에서 자름. EOS 없음. 토큰 수는 응답별 글자/토큰 비율로 추정
  - 첫 불일치가 4,095토큰 뒤면 건너뜀. `--cut_scope all`이면 불일치 표현 없는 잘린 응답도 사용
  - 결과: 230쌍 / 160문제 (건너뜀 127). 불일치 후 회복 chosen 71%, rejected 중 불일치 전 비율 중앙값 45%
- **자동 검사 (`check()`, 하나라도 어기면 중단)**
  - 모든 쌍이 최종 학습셋 문제이고 제외 목록(형식 87, 라벨 16)에 없음
  - 텍스트 안에 `<|im_end|>` 없음 (EOS는 학습 단계에서만 붙임)
  - 정답 쌍: chosen 정답 / rejected 오답 + 규칙 파싱 / 앞부분 없음
  - 루프 쌍(resample): rejected EOS 없음, 앞부분 있음, **rejected의 반복+재시작 비율 50% 이상이거나 원래 응답 끝까지 포함**
  - 루프 쌍(cut): rejected EOS 없음, 앞부분 없음, chosen 정답, **첫 보기 불일치가 rejected 안에 있음**
  - 문제당 개수 제한
- **확인 못 한 것**: RFT 학습 데이터와 chosen이 얼마나 겹치는지 (RFT 데이터가 정해지면 확인)

## [4] DPO 학습 (`scripts/train_dpo.py`, GPU)

- **모델**: `--model_path` 필수(RFT 병합 체크포인트 예정). LoRA라서 **기준 모델 = 같은 모델에서 어댑터를 끈 것 = 시작 모델**
- **입력 형식** (TRL 표준 텍스트 형식, 이미지 있음)
  - prompt = 평가 채팅 템플릿(이미지 먼저, 그다음 문제) + 앞부분
  - chosen = 이어쓰기 + `<|im_end|>`
  - rejected = 이어쓰기 + (정답 쌍만) `<|im_end|>`. 루프 쌍은 EOS 없음
  - TRL 1.17은 이미지 데이터셋에서 EOS를 자동으로 안 붙임(소스 확인) → 위 문자열 그대로 학습
- **이미지**: 평가와 같은 `qwen_vl_utils` 리사이즈를 미리 적용, 프로세서 크기 설정을 평가 값으로 고정
- **손실**: sigmoid DPO + chosen에 SFT 손실(가중치 1.0) = RPO 방식. beta 0.1. LD-DPO(`--ld_alpha`)는 옵션으로만 둠
- **LoRA**: GRPO와 같음 (LLM 디코더 전체 linear, r16, alpha 32, vision 동결, assert로 확인)
- **학습**: lr 5e-6(cosine, warmup 5%), 1에폭, 배치 1 × 누적 16, gradient checkpointing, bf16, `max_length` 없음(자르지 않음)
- **체크포인트**: 25스텝마다. 재개해도 이번 실행의 save/log 간격을 쓰도록 콜백에서 덮어씀 (GRPO run1에서 25/50/75 체크포인트가 안 남았던 문제)
- **`--check_only`** (실제 프로세서, CPU): 프로세서가 이미지를 다시 줄이지 않는지, `<|im_end|>`가 생성 종료 토큰인지, chosen/rejected 끝 토큰이 의도대로인지, 종류별 최대 길이(프롬프트+응답) 출력
- **TRL 제약**: 이미지 데이터셋은 기준 모델 확률 미리 계산(`precompute_ref_log_probs`) 불가 → 매 스텝 어댑터 끄고 한 번 더 계산 (시간 증가, 메모리는 모델 한 벌)
- **확인 못 한 것**: 메모리(가장 긴 쌍 = 프롬프트 + 앞부분 수천 토큰 + 이어쓰기 × chosen·rejected), 스텝 시간

---

## 실행 순서

`outputs/rft/merged`는 RFT 학습 후 병합한 모델 폴더를 뜻하는 **임시 이름**이에요. 실제 경로로 바꿔서 쓰세요.

### 1. 맥에서 (CPU) — 끝남

| 명령 | 하는 일 | 결과 파일 |
|---|---|---|
| `python scripts/loop_onset.py` | 잘린 응답 1,044개의 갈림길 후보(60/45/30/15%) 계산 | `outputs/dpo/loop_candidates.jsonl` |
| `python scripts/dpo_pairs.py` | 정답 쌍 531개 | `outputs/dpo/pairs.jsonl` |
| `python scripts/dpo_pairs.py --loop_mode cut --out outputs/dpo/pairs_cut.jsonl` | 정답 쌍 + 승희 방식 루프 쌍 230개 | `outputs/dpo/pairs_cut.jsonl` |

(`--compare_gold`는 정답지 비교용이고 결과는 `docs/loop_goldset_v1.md`에 정리돼 있어요. 다시 돌릴 필요 없음)

### 2. 서버로 파일 옮기기

`outputs/`와 `data/`는 git에 없어서 따로 보내야 해요. 위 결과 파일과 재료를 묶은 압축 파일을 레포 루트에서 풀면 같은 경로에 들어가요.

```bash
tar xzf dpo_server_bundle_YYYYMMDD.tar.gz        # 레포 루트에서
md5sum -c outputs/dpo/MANIFEST.md5                 # 전부 OK 나와야 함
```

들어 있는 것: 롤아웃 응답 원문(`rollouts_resp.jsonl`, `pilot_resp.jsonl`), `loop_candidates.jsonl`, `pairs.jsonl`, `pairs_cut.jsonl`, 문제 원문(`outputs/decontam/virl/clean.jsonl`), `data/rl/`의 학습·검증셋과 제외 목록 2개.
ViRL 이미지(`data/virl39k_raw`)는 들어 있지 않아요. 서버에서 받아야 해요: `hf download TIGER-Lab/ViRL39K --repo-type dataset --local-dir data/virl39k_raw`

### 3. 서버에서 (GPU, RFT 모델이 나온 뒤)

```bash
python scripts/test_dpo.py                                  # CPU 테스트 ("dpo pipeline OK")

# v0: 승희 방식 (재샘플링 없이 바로)
python scripts/train_dpo.py --model_path outputs/rft/merged --pairs outputs/dpo/pairs_cut.jsonl --check_only
python scripts/train_dpo.py --model_path outputs/rft/merged --pairs outputs/dpo/pairs_cut.jsonl --output_dir outputs/dpo/v0

# v1: 재샘플링 방식
python scripts/dpo_resample.py --model_path outputs/rft/merged --limit 20     # 스모크 테스트 (20개, 시간 확인)
python scripts/dpo_resample.py --model_path outputs/rft/merged                # 본 재샘플링 (오래 걸림)
python scripts/dpo_pairs.py --resample outputs/dpo/resample.jsonl --out outputs/dpo/pairs_resample.jsonl
python scripts/train_dpo.py --model_path outputs/rft/merged --pairs outputs/dpo/pairs_resample.jsonl --check_only
python scripts/train_dpo.py --model_path outputs/rft/merged --pairs outputs/dpo/pairs_resample.jsonl --output_dir outputs/dpo/v1

# 평가용 병합
python scripts/merge_lora.py --model_path outputs/rft/merged --adapter outputs/dpo/v0/final_adapter --out outputs/dpo/v0/merged
```

- `--check_only`: 학습 전에 이미지가 평가와 같게 들어가는지, chosen 끝에만 `<|im_end|>`가 붙는지, 가장 긴 쌍이 몇 토큰인지 출력
- 재샘플링과 DPO 모두 **RFT 모델로** 해요. DPO의 기준 모델이 시작 모델이라서, 쌍을 만든 모델과 시작 모델이 같아야 해요

## 열린 결정 (팀 논의)

- 남은 정답지 13개 라벨링 여부: G(루프 아님)가 나오는지 확인용. 지금 설계는 "잘림 = 전부 후보"라 G가 섞여도 rejected 자격(한도 안에 못 끝냄)은 있음
- 루프 쌍 chosen을 정답만으로 제한할지 (`--loop_chosen_correct`): 탈출 자체가 목표면 오답 탈출도 허용, 점수까지 노리면 정답만
- 정답 쌍과 루프 쌍 비율: 지금은 정답 쌍 531 대 루프 쌍 최대 615(재샘플링 대상 응답 수, 실제는 탈출한 것만). `--kinds`로 한쪽만 학습도 가능
- `--sft_weight`, beta, lr: 첫 실행은 기본값, 결과 보고 조정
