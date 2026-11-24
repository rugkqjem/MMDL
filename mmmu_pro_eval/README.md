# MMMU-Pro 성능 평가 (standard, 10지선다)

baseline(Qwen3-VL-4B-Instruct)과 우리가 학습한 v1b(RFT + DPO)를 **MMMU-Pro `standard (10 options)` test split 1,730문제**로 평가하고 비교합니다.

| 이름 | 모델 | 비고 |
|---|---|---|
| baseline | `Qwen/Qwen3-VL-4B-Instruct` @`ebb281ec` | MMMU-val 62.11 |
| v1b | [`yujin16/qwen3vl4b-mmmu-v1b`](https://huggingface.co/yujin16/qwen3vl4b-mmmu-v1b) | RFT + DPO(재샘플링 루프 쌍, lr 5e-5), LoRA 병합본. MMMU-val 64.89 ([결과 문서](../docs/rft_dpo_results.md)) |

## 실행

환경은 레포 루트 README의 MMMU 평가 환경(vLLM, `requirements.lock.txt`) 그대로입니다.

```bash
# GPU 2장: baseline은 GPU 0, v1b는 GPU 1에서 동시에 → 둘 다 끝나면 비교
bash mmmu_pro_eval/run_compare.sh --gpus 0,1 --data_root <HF datasets 캐시>
#   진행 로그: outputs/mmmu_pro_std10/{baseline,v1b}/run.log  (tail -f 로 확인)

# GPU 1장: 두 모델을 차례로
bash mmmu_pro_eval/run_compare.sh --data_root <HF datasets 캐시>

# 한 모델만 (예: baseline 먼저)
bash mmmu_pro_eval/run_compare.sh --data_root <캐시> --only baseline

# v1b를 로컬 병합본으로
bash mmmu_pro_eval/run_compare.sh --data_root <캐시> --v1b_path outputs/dpo/v1b/merged

# 빠른 확인 (앞 20문제만, 결과 폴더 따로)
bash mmmu_pro_eval/run_compare.sh --data_root <캐시> --limit 20 --out_root outputs/mmmu_pro_smoke
```

- 이미 `predictions.jsonl`이 있는 모델은 추론을 건너뛰고 채점만 다시 합니다. 다시 돌리려면 그 파일을 지우세요.
- 그 밖의 인자(`--max_new_tokens`, `--gpu_memory_utilization` 등)는 두 모델 추론에 똑같이 들어갑니다.
- 시간: MMMU-val 900문제가 A100 80GB에서 약 46분이었으니, 1,730문제는 모델당 약 1.5시간으로 예상합니다. `--gpus 0,1`이면 두 모델이 동시에 돌아서 전체도 약 1.5시간입니다.

CPU 테스트 (모델·데이터 없이): `python mmmu_pro_eval/test_mmmu_pro_eval.py`

## 결과 파일

```
outputs/mmmu_pro_std10/
  baseline/  predictions.jsonl, predictions_meta.json, scores_{qwen,mmmu}[_raw].{json,md,csv}
  v1b/       (같은 구성)
  compare/   compare.md (전체·과목별 표), compare.json, compare_per_question.csv
```

`compare.md`에 들어가는 것:
- 전체 정확도: **micro(1,730문제 전체 정답률, MMMU-Pro 공식 방식)**와 과목 평균(macro)
- 잘림 수(`finish_reason == length`), 파싱 실패 수
- 같은 문제끼리 비교: v1b만 맞힘 / baseline만 맞힘, 잘림 고침 / 새로 생김, 각각 정확 부호 검정 p값
- 과목별 정확도와 차이, 과목별 잘림 수

보고 점수는 MMMU-val과 같은 `scores_qwen` (최종 답 추출 → Qwen 규칙 파서, 실패는 오답)입니다. 다른 채점기로 비교하려면 `python mmmu_pro_eval/compare.py --scores scores_mmmu ...`.

## 평가 설정

| 항목 | 값 |
|---|---|
| 데이터 | HF `MMMU/MMMU_Pro`, config `standard (10 options)`, split `test` (1,730문제, 정답 공개). `--pro_revision`으로 고정 가능 |
| 프롬프트 | `{question}\nA. ...\n...\nJ. ...\n` + 공식 CoT 지시문 (`scripts/infer_mmmu_pro.py`의 `COT['standard']`, 출처 MMMU-Benchmark `mmmu-pro/prompts.yaml`) |
| 이미지 배치 | 기본 `--image_layout interleave`: 질문·보기 속 `<image n>` 자리에 `image_n`을 끼워 넣음 (공식 MMMU-Pro 추론 코드 방식). 같은 이미지가 두 번 나오면 처음 자리에만 넣고, 자리표시자가 없는 이미지는 텍스트 앞에 둠. `front`로 바꾸면 모든 이미지를 텍스트 앞에 둠 (`scripts/infer_mmmu_pro.py`, MMMU-val 평가와 같은 방식) |
| 생성 | MMMU-val과 동일 (`scripts/infer.py` 기본값): temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, seed 3407, max_new_tokens 16384, max_model_len 32768, bf16 |
| 이미지 해상도 | min_pixels 1280·28·28, max_pixels 5120·28·28 |
| 답 추출 | 마지막 `Answer: X` / `answer is` / `\boxed{}` → A–J 글자. 그다음 Qwen 규칙 파서 (`scripts/score.py`) |

### `scripts/infer_mmmu_pro.py`와의 관계

`rl-grpo` 브랜치의 `scripts/infer_mmmu_pro.py`(`run_mmmu_eval.sh --pro standard`)와 프롬프트·생성 설정·출력 형식이 같고, **이미지 배치만 다릅니다** (그쪽은 이미지를 모두 앞에 둠). `--image_layout front`로 돌리면 그 스크립트와 같은 입력이 됩니다. 이 폴더는 여기에 두 모델 비교, 과목별 표, 같은 문제끼리 검정을 더한 것입니다.

### 주의

- 학습 데이터는 `scripts/decontaminate.py`로 MMMU-Pro(standard 10지선다 + vision)와의 중복을 제거한 것입니다.
- 평가는 샘플링(temperature 0.7)이라 실행마다 흔들립니다. MMMU-val에서 같은 설정 재실행 차이가 약 0.7점이었으니, 1점 안팎 차이는 같은 문제끼리 비교한 p값과 함께 보세요.
