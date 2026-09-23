# Qwen3-VL-4B MMMU-val 베이스라인 정리 (2026-09-23)

팀 회의 전 정리 노트. 오늘 `assignment1-vllm-eval` 브랜치 파이프라인으로 900문제를 처음 끝까지 돌린 결과와 결정 사항. 제출용 보고서(`reports/mmmu_baseline.md`, `docs/SUBMISSION_TEMPLATE.md` 형식)는 회의 후 이 내용을 바탕으로 작성.

## 한눈에

| | 점수 (macro, %) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| **우리: Qwen 파서 + 최종 답 추출 (주 지표)** | **62.11** (559/900, Δ −5.29) |
| 우리: MMMU 파서 + 최종 답 추출 | 62.22 |
| 참고: 추출 없이 원본 파서에 응답 전체 입력 | Qwen 32.67 / MMMU 50.78 |
| 참고: 이전 `assignment1-eval` 브랜치 ("글자만 출력", 128 토큰, Text recipe) | 53.89 |

- 끝까지 생성된 815개만 보면 정확도 **67.73%**로 공식과 거의 같음. 격차의 대부분은 16384 토큰에서 잘린 85개(9.4%)에서 나옴.
- 결과 파일: `outputs/qwen3vl4b_mmmu_val/` (`predictions.jsonl` 원본 응답, `predictions_meta.json`, `scores_*`)

## 실행 환경

| 항목 | 값 |
|---|---|
| 모델 / 데이터 | `Qwen/Qwen3-VL-4B-Instruct` @ebb281ec (bf16) / HF `MMMU/MMMU` @98e6ac0c, validation, 30과목 × 30 |
| 백엔드 | vLLM 0.30.0, transformers 5.17.0, torch 2.13.0+cu130, Python 3.12.14 (`requirements.lock.txt`) |
| GPU | A100-SXM4-80GB × 1 |
| 시간 | `infer.py` 전체 46.3분 (생성 42.1분). 채점은 CPU 수 초 |
| VRAM | nvidia-smi 최대 75.6GB. 대부분 vLLM이 미리 잡은 KV cache(59.9 GiB)이고, 가중치 등 9.0 GiB + activation 2.4 GiB |

```bash
bash scripts/run_mmmu_eval.sh --model_path Qwen/Qwen3-VL-4B-Instruct \
  --data_root <MMMU HF datasets cache> --output_dir outputs/qwen3vl4b_mmmu_val
```

- venv는 **헤더(`Python.h`)가 있는 Python 3.12**가 필요. vLLM 엔진이 뜰 때 Triton이 C 확장을 컴파일함. 서버의 시스템 3.12에는 헤더가 없어 uv 관리 Python을 씀 (`uv venv --seed --python 3.12 .venv`)
- `load_dataset`은 validation만 써도 과목별 test split까지 받음 (첫 실행 때 약 3GB)
- 900개 요청을 `llm.generate()` 한 번에 넣음 (continuous batching). 응답 길이가 중앙값 450 ~ 최대 16384 토큰으로 편차가 커서 이 방식이 유리

## 파이프라인 설정

**프롬프트**: Qwen 공식 `evaluation/mmmu/run_mmmu.py` @96588727 `build_mmmu_prompt()`와 같음 (tech report Appendix B.1과도 같음). 시스템 프롬프트 없이 이미지를 모두 텍스트 앞에 둠.
```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: {question}
Options:
A. {option_A}
B. {option_B}
...
Please select the correct answer from the options above.<|im_end|>
<|im_start|>assistant
```
주관식(53문제)은 `Question: {question}`만 들어감.

**샘플링**: 모델 카드 @ebb281ec "Generation Hyperparameters"의 **VL** recipe (`generation_config.json`과 같은 값)
- temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5
- seed 3407 (GitHub README 값. 공식 `run_mmmu.py`는 42)

**생성 예산**: `max_new_tokens` 16384 (모델 카드 `out_seq_length`), `max_model_len` 32768
- 2048 초과 240개(26.7%), 8192 초과 115개(12.8%), 16384에서 잘림 85개(9.4%)
- 2048로는 약 27%가 잘려서 제외. 예산을 더 늘리면 잘림은 줄겠지만 긴 응답 몇 개가 전체 시간을 좌우함 (KV cache 43.6만 토큰)

**이미지**: `min_pixels = 1280·28·28`, `max_pixels = 5120·28·28` (공식 코드 값)
- 이 상수는 Qwen2.5-VL의 28px 기준. Qwen3-VL은 32px(patch 16 × merge 2)당 시각 토큰 1개라서 실제로는 이미지당 약 980~3,900 토큰
- 982장 모두 `min_pixels` 이상, 즉 작은 이미지는 약 1MP로 **확대**됨. 공식 파이프라인과 조건을 맞추려고 유지

## 채점 방식

`scripts/score.py`: **최종 답 추출 → 공개 파서 원본(수정 없음)**. 체크포인트만 바꿔 다시 돌릴 때도 같은 규칙이 적용되도록, 추출 규칙은 다른 벤치마크에서 쓰이는 공개 규칙만 사용.

1. **최종 답 추출** (`--extract final`, 기본값): 응답에서 가장 뒤에 나오는 표시를 찾음
   - 마지막 `\boxed{…}` — hendrycks/math `last_boxed_only_string`
   - 마지막 `answer is` / `Answer:` 뒤의 텍스트 — MMLU-Pro `extract_answer`/`extract_again`
   - 판단이 들어간 부분: 마크다운 `**`, `$` 제거와 `answer is:`(콜론)도 허용
   - 객관식은 추출 결과가 `C`, `C.`, `(C)`로 시작하면 그 글자만 넘김
   - **표시가 없으면 응답 전체를 넘김** (원본 방식과 같음)
2. **Qwen 파서** (주 지표): Qwen 공식 `eval_utils.py`의 `can_infer` (VLMEvalKit 계열). 주관식은 `{A: 정답, B: "Other Answers"}` 2지선다로 판정. 공식은 규칙이 실패하면 GPT-3.5 judge를 쓰지만 **우리는 judge 없이 실패 = 오답**
3. **MMMU 파서** (보조): MMMU 공식 `eval_utils.py`. 답을 못 찾으면 seed 42로 **무작위 선택**
- `--extract none`: 추출 없이 응답 전체를 파서에 넣는 원본 방식 → `scores_{parser}_raw.*`

**왜 추출이 필요했나**: 모델이 평균 2,846 토큰의 풀이를 쓰고 끝에 `Final Answer: **C. …**`처럼 답함. 풀이 전체를 넣으면 여러 보기 글자가 나와서 Qwen 파서는 504개 실패, MMMU 파서는 엉뚱한 글자를 고르는 경우가 많았음.

| | Qwen 파서 | MMMU 파서 |
|---|---|---|
| 추출 적용 | 62.11 (실패 96) | 62.22 (무작위 78) |
| 원본 방식 | 32.67 (실패 504) | 50.78 (무작위 63) |

**Qwen 파서를 주 지표로 정한 이유**
- 끝까지 생성됐지만 표시 없이 `D. Organic`처럼 짧게 답한 객관식 42개를 MMMU 파서는 37개 무작위 처리함. ` D `처럼 공백으로 둘러싸인 글자만 인식하고, 5단어 이하면 보기 내용 비교도 하지 않기 때문
- 같은 42개를 Qwen 파서는 40개 정상 처리함. fine-tuning으로 답이 짧아지면 MMMU 파서만 불리해질 수 있음

## 과목별 결과

| No. | Subject | Qwen | MMMU | 잘림 |
|---|---|---|---|---|
| 1 | Accounting | 73.33 | 80.00 | 5 |
| 2 | Agriculture | 53.33 | 43.33 | 0 |
| 3 | Architecture_and_Engineering | 46.67 | 63.33 | 12 |
| 4 | Art | 60.00 | 43.33 | 3 |
| 5 | Art_Theory | 86.67 | 80.00 | 1 |
| 6 | Basic_Medical_Science | 73.33 | 66.67 | 0 |
| 7 | Biology | 50.00 | 50.00 | 2 |
| 8 | Chemistry | 33.33 | 36.67 | 1 |
| 9 | Clinical_Medicine | 60.00 | 60.00 | 0 |
| 10 | Computer_Science | 60.00 | 60.00 | 0 |
| 11 | Design | 76.67 | 66.67 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30.00 | 26.67 | 0 |
| 13 | Economics | 76.67 | 76.67 | 0 |
| 14 | Electronics | 46.67 | 43.33 | 5 |
| 15 | Energy_and_Power | 56.67 | 60.00 | 12 |
| 16 | Finance | 60.00 | 70.00 | 3 |
| 17 | Geography | 43.33 | 50.00 | 2 |
| 18 | History | 70.00 | 70.00 | 0 |
| 19 | Literature | 83.33 | 73.33 | 0 |
| 20 | Manage | 66.67 | 70.00 | 0 |
| 21 | Marketing | 83.33 | 83.33 | 0 |
| 22 | Materials | 53.33 | 60.00 | 11 |
| 23 | Math | 56.67 | 53.33 | 1 |
| 24 | Mechanical_Engineering | 43.33 | 56.67 | 13 |
| 25 | Music | 30.00 | 40.00 | 11 |
| 26 | Pharmacy | 76.67 | 76.67 | 2 |
| 27 | Physics | 76.67 | 76.67 | 0 |
| 28 | Psychology | 80.00 | 76.67 | 0 |
| 29 | Public_Health | 90.00 | 90.00 | 1 |
| 30 | Sociology | 66.67 | 63.33 | 0 |
| | **Overall (macro avg)** | **62.11** | **62.22** | **85** |

- Overall = 30개 과목 정확도의 단순 평균. 과목당 30문제라 micro(559/900)와 같음
- 잘림이 많은 과목(건축, 에너지, 재료, 기계, 음악)은 MMMU 점수가 더 높음. 답이 없는 잘린 응답을 MMMU 파서가 무작위로 골라 일부 맞힌 몫

## 공식 수치(67.4)와의 격차: 근거

1. **잘림 (주원인)**
   - 85개(9.4%)가 16384 토큰에서 끊겨 최종 답이 없음 → 7개만 정답
   - 끝까지 생성된 815개의 정확도는 67.73%
   - 잘림이 5개 이상인 7과목의 macro 50.0, 나머지 23과목 65.8
   - 잘린 85개 중 18개는 "Wait — perhaps…" 같은 문장 반복 루프
   - 단, 잘린 문제는 원래 어려운 문제일 수 있어서 67.73은 상한 추정치
2. **judge 생략**
   - 끝까지 생성됐는데 Qwen 규칙이 실패한 38개 (다 맞았다고 쳐도 최대 +4.2점)
   - 그중 32개가 주관식이고, `\dfrac{1}{64}` vs `1/64`, `2\sqrt{2}` vs `2.83` 같은 표기 차이가 섞여 있음. 공식 GPT judge라면 일부는 정답 처리했을 것
3. **파싱 방식의 민감도**: 같은 응답인데 추출 유무로 32.67 ↔ 62.11. 풀이형 응답에서는 채점 방식이 점수를 좌우함
4. **샘플링 변동**: temperature 0.7로 한 번만 실행했으므로 표준오차 약 ±1.6점
5. **데이터 소스**: 공식은 VLMEvalKit `MMMU_DEV_VAL.tsv`, 우리는 HF pinned 데이터. 이미지 인코딩이 다를 수 있음 (미검증)

## 회의에서 정할 것
- **주 지표를 Qwen 파서 + 추출로 확정할지** (MMMU 파서는 보조로 같이 보고)
- **`max_new_tokens`**: 16384 유지(모델 카드 값)인지, 잘림을 줄이려고 늘릴지. 늘리면 공식 recipe에서 벗어나고 시간이 늘어남
- **judge**: 규칙 실패분에만 LLM judge를 쓸지. 재현성과 비용 문제가 있음
- **반복 실행**: seed를 바꿔 점수가 얼마나 흔들리는지 측정할지 (1회 약 45분)
- 제출 보고서 분담, 팀명

## 확인 못 한 것
- 24GB GPU(RTX 4090)에서 실행: 가중치는 bf16으로 약 9GB라 `--gpu_memory_utilization`, `--max_model_len`을 낮추면 될 것으로 보지만 미검증
- 공식 TSV와 HF 데이터의 이미지 차이
