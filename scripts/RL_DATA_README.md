# GRPO 학습 데이터 준비 (ViRL39K → 중복 제거 → 난이도 필터링 → 학습셋)

이 파일들은 레포의 `scripts/`에 넣고 **레포 루트에서** 실행합니다. 평가 코드를 다시 구현하지 않고 그대로 import해요.

| 파일 | 하는 일 | 가져다 쓰는 평가 코드 |
|---|---|---|
| `rl_prepare_virl39k.py` | ViRL39K parquet → qwen-vl-finetune JSONL | `infer.build_messages` (평가와 같은 프롬프트) |
| `decontaminate.py` | MMMU dev/val/test + MMMU-Pro 중복 제거 | `infer.SUBJECTS`, `DATASET_REVISION` |
| `rl_rollout_filter.py` | 문제당 n=8 샘플링 → pass rate 기록 | `infer.prepare_sample`, `make_llm`, `make_sampling`, `resample_wrong.is_correct` |
| `rl_select.py` | 학습 신호가 있는 문제 선별 + 리포트 | `score_qwen`이 그대로 쓸 수 있는 필드 출력 |

평가와 완전히 같은 것: 프롬프트(rollout 중 assert로 확인), 이미지 min/max pixels, 모델 revision, VL 샘플링 레시피, 채점
(`extract_final_answer` + `score_qwen`).
일부러 다르게 한 것 두 가지:
- `--max_new_tokens` 기본값이 8192예요(평가는 16384). **RL 학습 때 쓸 최대 응답 길이와 같게 맞추세요.**
- 잘린 응답은 `correct`에서 오답으로 셉니다(RL 보상의 잘림 페널티와 맞추기 위해). 평가 기준으로 맞았는지는 `n_correct_eval`에 따로 남겨요.

## 0. 의존성
기존 평가 환경(vllm, transformers, qwen-vl-utils, datasets)에 `pandas pyarrow datasketch imagehash`를 추가하세요.

## 1. 다운로드 + 변환
```bash
hf download TIGER-Lab/ViRL39K --repo-type dataset --local-dir data/virl39k_raw
python scripts/rl_prepare_virl39k.py --src data/virl39k_raw --out data/virl39k/all.jsonl --dump_skipped 15  # 카테고리 분포 확인
python scripts/rl_prepare_virl39k.py --src data/virl39k_raw --out data/virl39k/sub.jsonl --max_per_category 3000
```
ViRL39K는 보기를 문제 텍스트 안에 여러 형식으로 적어 둬요("Choices: (A) .. (B) ..", 줄마다 "A. ..", "A: ..", 헤더 없이 "(A) .. (B) .."
등). 변환기가 A, B, C… 순서로 이어지는 표시를 찾아 보기로 떼어 내고, 정답이 글자든 보기 내용이든 해당 글자로 바꿔요.
`--dump_skipped N`을 주면 버려진 문제를 이유별로 N개씩 `<out>_skipped.txt`에 남겨서, 변환기가 놓치는 형식을 바로 확인할 수 있어요.
GradeSchool 카테고리는 미리 빼지 않아요. 너무 쉬운 문제는 난이도 필터링에서 어차피 걸러져요.
변환된 행의 `passrate_32b`는 원본의 `PassRate_32BTrained`(RL 학습된 32B 모델의 정답률)예요. 값이 없으면 -1이에요.
**객관식만 남는 게 기본값이에요.** `score_qwen`은 주관식을 `{A: 정답, B: 'Other Answers'}`로 바꾼 뒤
"예측에 정답 문자열이 포함돼 있는지"로 채점해요. 그래서 정답이 "3"이면 "13"도 정답 처리됩니다.
점수를 보고할 때는 괜찮지만, RL 보상으로 쓰면 모델이 이 허점을 파고들 수 있어요(보상 해킹). MMMU-val도 대부분 객관식이고요.
정답은 글자인데 보기 목록을 찾지 못한 문제(`mc_unparsed`)와, 보기는 찾았지만 정답이 어느 보기와도 안 맞는 문제(`mc_gold_not_in_options`)는
`score_qwen`으로 채점할 수 없어서 뺍니다.

## 2. MMMU 중복 제거
```bash
python scripts/decontaminate.py --data data/virl39k/sub.jsonl --image_root data/virl39k_raw --out outputs/decontam/virl
# review.html 확인 → decisions.csv에서 오탐만 keep으로 바꾼 뒤
python scripts/decontaminate.py --data data/virl39k/sub.jsonl --out outputs/decontam/virl --apply
```
이번에 바뀐 점: 행에 `question`/`options` 필드가 있으면, MMMU 쪽과 같은 방식(문항 + 보기)으로 텍스트를 비교해요.
평가 프롬프트의 고정 문구가 섞이면 짧은 문제에서 MinHash 유사도가 낮게 나와 중복을 놓칠 수 있기 때문이에요.
단순한 도형이나 차트 이미지는 pHash가 오탐을 낼 수 있으니 review.html에서 꼭 확인하세요.

## 3. 파일럿 (200문제)
```bash
python scripts/rl_rollout_filter.py --data outputs/decontam/virl/clean.jsonl --image_root data/virl39k_raw \
    --out outputs/rl_rollout/pilot.jsonl --limit 200 --shuffle --save_responses outputs/rl_rollout/pilot_resp.jsonl
```
- 로그의 `tok/s`, `q/h`로 전체 실행 시간을 계산하세요: `문제 수 ÷ q/h`.
- `pilot_resp.jsonl`에서 `correct_eval=false`인 응답 20개 정도를 직접 읽어 보세요. 실제로는 맞았는데 채점이 놓친 경우가 있는지 확인하는 거예요.
  ViRL 문제에서도 모델이 "answer is"/"Answer:" 형식으로 답하는지가 핵심이에요(MMMU에서는 900개 중 743개가 그랬어요).

## 4. 전체 실행 (끊겨도 같은 명령으로 이어서 실행)
```bash
nohup python scripts/rl_rollout_filter.py --data outputs/decontam/virl/clean.jsonl --image_root data/virl39k_raw \
    --out outputs/rl_rollout/rollouts.jsonl > rl_rollout.log 2>&1 &
```
처음 4번이 전부 맞으면 "너무 쉬움"으로 처리하고 나머지 4번은 건너뛰어요(`--stage1_n 0`이면 처음부터 8번).

## 5. 학습셋 선별
```bash
python scripts/rl_select.py --rollouts outputs/rl_rollout/rollouts.jsonl --data outputs/decontam/virl/clean.jsonl \
    --out data/rl/train.jsonl --image_root data/virl39k_raw --keep_trunc_mixed --val_size 300
```
- 기본 규칙: 8번 중 1~7번 맞힌 문제만 남겨요.
- `--keep_trunc_mixed`: 0번 맞혔더라도 일부만 잘린 문제를 같이 남겨요. 잘림 페널티가 있는 보상이면 이 그룹도 학습 신호가 있어요.
- `train_report.md`에 버킷 분포, 카테고리별 pass rate, 잘림 비율이 정리돼요.
- 출력의 `problem`은 `<image>` 태그 + 평가 프롬프트예요(EasyR1 형식). EasyR1을 쓰면 기본 `format_prompt`(r1v.jinja 같은 수학용 템플릿)를
  **꺼야** 평가 프롬프트가 그대로 유지돼요. 보상 함수는 6번의 `rl_reward.py`를 쓰세요.

## 6. 보상 함수 (`rl_reward.py`)
평가 채점기(`extract_final_answer` + `score_qwen`)를 그대로 쓰고 잘림 패널티만 더해요. 잘림을 **채점보다 먼저** 확인해요.

| 응답 | 보상 |
|---|---:|
| 잘림 (최대 길이에 닿았고 마지막 토큰이 EOS가 아님) | `-trunc_penalty` (기본 0.5) |
| 끝까지 쓰고 정답 | 1.0 |
| 끝까지 쓰고 오답 / 파싱 실패 | 0.0 |

- TRL `GRPOTrainer`용: `make_reward_func(max_completion_length=8192, eos_ids=..., trunc_penalty=0.5)`. 데이터셋에 `answer`, `options` 열이 있어야 해요.
  `GRPOConfig(mask_truncated_completions=False)`로 두세요(True면 잘린 응답이 학습에서 빠져 루프 신호가 사라져요).
- 주관식(보기 없음)은 오류를 내요. `score_qwen`이 "정답 문자열 포함"으로 채점해서 보상 해킹이 되기 때문이에요.
- 검증(CPU):
```bash
python scripts/test_rl_reward.py                     # 합성 케이스
python scripts/check_rl_reward.py --mmmu             # MMMU 객관식 847개: 평가 채점과 847/847 일치해야 함
python scripts/check_rl_reward.py --rollouts_resp outputs/rl_rollout/rollouts_resp.jsonl \
    --ids data/rl/train_noA.jsonl --penalties 0 0.25 0.5 1   # 롤아웃 판정 재현 + 패널티별 '신호 없는 그룹' 비율
```

## GPU 없이 파이프라인만 테스트
```bash
python scripts/rl_rollout_filter.py --data data/virl39k/sub.jsonl --image_root data/virl39k_raw --out /tmp/t.jsonl --limit 50 --dry_run
```
