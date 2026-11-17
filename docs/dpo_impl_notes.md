# DPO 파이프라인 구현 명세 (코드 기준)

> 2026-11-17 · 브랜치 `rl-grpo` · **코드가 실제로 하는 일**만 적은 문서예요. 주의점 조사 문서와 항목별로 대조하고, 서버에서 돌릴 때 순서를 보는 용도예요.
> GPU 단계(재샘플링, 학습)는 아직 한 번도 안 돌렸어요.

## 전체 흐름

```
롤아웃 응답 (원본 모델, ViRL, 평가 샘플링, 8192토큰)
  ├─ [1] loop_onset.py      잘린 응답마다 갈림길 후보 계산                        CPU
  ├─ [2] dpo_resample.py    갈림길에서 다시 이어쓰기 → 빠져나오는 지점 찾기         GPU (RFT 모델로)
  ├─ [3] dpo_pairs.py       정답 쌍 + 루프 쌍 만들기, 규칙 검사                   CPU
  └─ [4] train_dpo.py       TRL DPOTrainer + LoRA                               GPU
```

- 문제 범위: **최종 학습셋 `data/rl/train_final.jsonl`(780문제)**만. 검증셋 272문제는 안 씀
- 루프 쌍은 두 방식이 있어요. **승희 방식(cut)**은 [1][2] 없이 [3]에서 바로 만들고, **재샘플링 방식**은 [1][2]를 거쳐요. 두 방식 비교는 `docs/dpo_loop_pairs_compare.md`

---

## [1] 갈림길 후보 (`scripts/loop_onset.py`)

- **입력 → 출력**: 롤아웃 응답 원문 → `outputs/dpo/loop_candidates.jsonl` (잘린 응답마다 한 줄)
- **대상**: 잘린 응답(EOS 없이 8192토큰 도달) **1,044개 전부**. 루프인지 따로 거르지 않음
  - 이유: 사람이 라벨한 잘린 응답 17개가 전부 루프였음
- **갈림길 후보**: 응답 길이의 **60 / 45 / 30 / 15%** 지점에서, 그 줄의 시작 위치. 늦은 것부터
  - 루프 시작 위치를 텍스트로 찾지 않는 이유: 텍스트 규칙(줄 반복 비율, 망설임 표현 밀도)으로 찾은 위치가 사람 라벨과 잘 맞지 않았음 → 위치는 [2]에서 실제로 다시 생성해 보고 정함
- 텍스트 규칙 결과(반복 규칙, 밀도 규칙, 기존 규칙)는 파일에 **기록만** 해요. 거르는 데는 안 써요

## [2] 갈림길 재샘플링 (`scripts/dpo_resample.py`, GPU)

- **입력 → 출력**: `loop_candidates.jsonl` → `outputs/dpo/resample.jsonl` (+ `_meta.json`에 샘플링 설정 기록)
- **모델**: `--model_path` 필수. **RFT 병합 체크포인트**로 돌려요 (DPO를 시작할 모델과 같아야 해서)
- **대상**: 문제당 잘린 응답 최대 2개 → **615개 응답, 375문제**
- **하는 일**: 평가와 같은 프롬프트(`infer.prepare_sample`) + 원래 응답의 갈림길까지 앞부분을 주고, **8번 이어쓰기**
  - 샘플링: 평가 설정 그대로 (temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5)
  - 최대 토큰 = 8192 − 앞부분 토큰 수 (롤아웃과 같은 총 예산). 남은 예산이 512토큰 미만이면 건너뜀
  - **탈출** = 이어쓰기가 EOS로 끝남. 정답 여부도 평가 채점기로 기록
- **갈림길 고르는 순서**: 60% 지점부터. 하나라도 탈출하면 그 지점으로 확정. 전부 다시 잘리면 한 단계 앞 지점으로 → **"실제로 빠져나올 수 있는 가장 늦은 지점"**
- **확인 못 한 것**: 실제 탈출률, 소요 시간 (최대 615개 × 8번 × 4라운드)

## [3] DPO 쌍 (`scripts/dpo_pairs.py`, CPU)

**정답 쌍** (문제당 1개) — 531쌍 / 531문제
- chosen = EOS로 끝난 정답 응답
- rejected = EOS로 끝난 오답 응답. 답이 규칙으로 파싱된 것만 (파싱 실패 제외)
- 두 응답 길이 1.5배 이내, 그중 가장 비슷한 쌍 (길이 비율 중앙값 1.06)

**루프 쌍, 승희 방식** (`--loop_mode cut`, 문제당 최대 2개) — 230쌍 / 160문제
- chosen = 같은 문제의 다른 정답 응답. "보기 불일치 후 회복" 응답 우선 (71%)
- rejected = "보기와 안 맞는다" 표현이 있는 잘린 응답을 **처음부터**, max(chosen 길이, 첫 불일치 + 1,500토큰), 최대 4,095토큰에서 줄바꿈 기준으로 자름. EOS 없음
- 첫 불일치가 4,095토큰보다 뒤면 건너뜀 (127개)

**루프 쌍, 재샘플링 방식** (`--resample`, 문제당 최대 2개) — [2] 이후에 만들 수 있음
- 앞부분 = 원래 응답의 갈림길까지. **프롬프트 쪽에 들어가서 손실 계산에서 빠짐**
- chosen = 그 갈림길에서 탈출한 이어쓰기 (정답 우선, `--loop_chosen_correct`면 정답만)
- rejected = 같은 갈림길 이후의 원래 응답. EOS 없음. chosen 이상, 최소 2,000자. 반복·망설임 텍스트가 50% 이상 될 때까지 늘림 (원래 응답 끝에서 멈춤)

**자동 검사** (하나라도 어기면 중단)
- 모든 쌍이 최종 학습셋 문제이고, 제외 목록(형식 87, 라벨 16)에 없음
- 텍스트 안에 `<|im_end|>` 없음 (EOS는 [4]에서만 붙임)
- 정답 쌍: chosen 정답 / rejected 규칙 파싱된 오답
- 승희 방식: chosen 정답, rejected EOS 없음, **첫 보기 불일치가 rejected 안에 있음**
- 재샘플링 방식: rejected EOS 없음, **반복·망설임 50% 이상이거나 원래 응답 끝까지 포함**
- 문제당 개수 제한

## [4] DPO 학습 (`scripts/train_dpo.py`, GPU)

- **모델**: `--model_path` 필수 (RFT 병합 체크포인트). LoRA라서 **기준 모델 = 어댑터를 끈 같은 모델 = RFT 모델**
- **입력 형식** (TRL 문자열 형식, 이미지 포함)
  - prompt = 평가 채팅 템플릿(이미지 먼저, 그다음 문제) + 앞부분
  - chosen = 응답 + `<|im_end|>`
  - rejected = 응답 + `<|im_end|>`(정답 쌍만). **루프 쌍은 EOS 없음**
  - TRL 1.17은 이미지 데이터셋에서 EOS를 자동으로 안 붙임 (소스 확인) → 위 문자열 그대로 학습
- **이미지**: 평가와 같은 `qwen_vl_utils` 리사이즈를 미리 적용, 프로세서 크기를 평가 값으로 고정
- **손실**: sigmoid DPO + chosen에 SFT 손실(가중치 1.0), beta 0.1. LD-DPO(`--ld_alpha`)는 옵션
- **LoRA**: LLM 디코더 전체 linear, r16, alpha 32, vision 동결 (assert로 확인)
- **학습**: lr 5e-6 (cosine, warmup 5%), 1에폭, 배치 1 × 누적 16, gradient checkpointing, bf16, 길이 자르지 않음
- **체크포인트**: 25스텝마다. 이어서 학습해도 이번 실행의 저장 간격을 씀 (GRPO run1에서 체크포인트가 안 남았던 문제 수정)
- **`--check_only`**: 학습 없이 이미지가 평가와 같게 들어가는지, chosen 끝에만 `<|im_end|>`가 붙는지, 가장 긴 쌍이 몇 토큰인지 출력
- **TRL 제약**: 이미지 데이터셋은 기준 모델 확률을 미리 계산 못 함 → 매 스텝 한 번 더 계산 (시간 증가, 메모리는 모델 한 벌)
- **확인 못 한 것**: 메모리, 스텝 시간

---

## 서버에서 실행하기

`outputs/rft/merged`는 RFT 학습 후 병합한 모델 폴더를 뜻하는 **임시 이름**이에요. 실제 경로로 바꿔서 쓰세요.

### 1. 데이터 받기

`outputs/`와 `data/`는 git에 없어요. [1]과 [3] 결과, 롤아웃 응답, 학습셋을 묶은 압축 파일(`dpo_server_bundle_YYYYMMDD.tar.gz`)을 레포 루트에서 풀어요.

```bash
tar xzf dpo_server_bundle_YYYYMMDD.tar.gz
md5sum -c outputs/dpo/MANIFEST.md5                                    # 전부 OK
hf download TIGER-Lab/ViRL39K --repo-type dataset --local-dir data/virl39k_raw   # 이미지 (압축 파일에 없음)
python scripts/test_dpo.py                                            # "dpo pipeline OK"
```

### 2. v0: 승희 방식 (RFT 끝나면 바로)

```bash
python scripts/train_dpo.py --model_path outputs/rft/merged --pairs outputs/dpo/pairs_cut.jsonl --check_only
python scripts/train_dpo.py --model_path outputs/rft/merged --pairs outputs/dpo/pairs_cut.jsonl --output_dir outputs/dpo/v0
python scripts/merge_lora.py --model_path outputs/rft/merged --adapter outputs/dpo/v0/final_adapter --out outputs/dpo/v0/merged
```

### 3. v1: 재샘플링 방식

```bash
python scripts/dpo_resample.py --model_path outputs/rft/merged --limit 20      # 20개만 먼저 (에러·시간 확인)
python scripts/dpo_resample.py --model_path outputs/rft/merged                 # 본 재샘플링 (오래 걸림)
python scripts/dpo_pairs.py --resample outputs/dpo/resample.jsonl --out outputs/dpo/pairs_resample.jsonl
python scripts/train_dpo.py --model_path outputs/rft/merged --pairs outputs/dpo/pairs_resample.jsonl --check_only
python scripts/train_dpo.py --model_path outputs/rft/merged --pairs outputs/dpo/pairs_resample.jsonl --output_dir outputs/dpo/v1
python scripts/merge_lora.py --model_path outputs/rft/merged --adapter outputs/dpo/v1/final_adapter --out outputs/dpo/v1/merged
```

---

## 열린 결정 (팀 논의)

- **E·G 제외**: 승희 방식 문서는 E(판독 재시도)·G(정상적인 긴 계산)를 빼자고 함. 두 방식 모두 아직 미적용 → 승희의 "의미 밀도 4.5" 정의를 받으면 필터 추가
- **학습 설정이 승희 방식 문서와 다른 것**: sft 항 (코드 1.0 / 문서는 기본 끄고 비교), lr (5e-6 / 1e-5), warmup (5% / 10%) → 하나로 정하기
- **재샘플링 chosen을 정답만으로 제한할지** (`--loop_chosen_correct`): 빠져나오는 것 자체가 목표면 오답도 허용, 점수까지 노리면 정답만
- **RFT 데이터와 DPO chosen 겹침**: RFT 데이터가 정해지면 확인
