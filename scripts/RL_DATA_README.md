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
huggingface-cli download TIGER-Lab/ViRL39K --repo-type dataset --local-dir data/virl39k_raw
python scripts/rl_prepare_virl39k.py --src data/virl39k_raw --out data/virl39k/all.jsonl      # 카테고리 분포 확인
python scripts/rl_prepare_virl39k.py --src data/virl39k_raw --out data/virl39k/sub.jsonl \
    --exclude_categories GradeSchool --max_per_category 3000
```
**객관식만 남는 게 기본값이에요.** `score_qwen`은 주관식을 `{A: 정답, B: 'Other Answers'}`로 바꾼 뒤
"예측에 정답 문자열이 포함돼 있는지"로 채점해요. 그래서 정답이 "3"이면 "13"도 정답 처리됩니다.
점수를 보고할 때는 괜찮지만, RL 보상으로 쓰면 모델이 이 허점을 파고들 수 있어요(보상 해킹). MMMU-val도 대부분 객관식이고요.
보기를 파싱할 수 없는 문제(정답은 글자인데 보기 형식이 특이한 경우)는 `score_qwen`으로 채점할 수 없어서 뺍니다.

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
  **꺼야** 평가 프롬프트가 그대로 유지돼요. 보상 함수는 `extract_final_answer` + `score_qwen` + 잘림 페널티로 따로 작성해야 합니다.

## GPU 없이 파이프라인만 테스트
```bash
python scripts/rl_rollout_filter.py --data data/virl39k/sub.jsonl --image_root data/virl39k_raw --out /tmp/t.jsonl --limit 50 --dry_run
```
