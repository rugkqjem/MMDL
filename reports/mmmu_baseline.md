# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 1조
- **팀원**: 김유진, 류승희, 송지훈
- **작성일**: 2026-09-27
- **재현 커맨드**: `bash scripts/run_mmmu_eval.sh --model_path Qwen/Qwen3-VL-4B-Instruct --data_root <MMMU HF cache> --output_dir outputs/qwen3vl4b_mmmu_val`

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17), bf16 |
| 추론 백엔드 | vLLM 0.30.0 (transformers 5.17.0, torch 2.13.0+cu130, Python 3.12.14)<br> ㄴ transformers.generate()는 배치 없이 문제당 1분 이상(900문제 15시간+) 걸려, continuous batching으로 900개를 한 번에 처리하는 vLLM 사용(생성 42분).|
| 사용 GPU | NVIDIA A100-SXM4-80GB × 1 |
| 실측 peak VRAM | 75.6 GB (nvidia-smi). 대부분 vLLM이 미리 잡은 KV cache(59.9 GiB)이며, 가중치 및 정적 메모리 9.0 GiB + activation 2.4 GiB |
| 총 소요 시간 | 46.3분 (`infer.py`).  그중 `llm.generate()` 생성 42.1분 |
| 의존성 | [`requirements.txt`](../requirements.txt) (최소 버전), [`requirements.lock.txt`](../requirements.lock.txt) (실제 실행 환경 고정) |
| 실행 커맨드 | ```bash bash scripts/run_mmmu_eval.sh --model_path Qwen/Qwen3-VL-4B-Instruct --data_root <MMMU HF datasets cache> --output_dir outputs/qwen3vl4b_mmmu_val``` |

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (이미지는 모두 텍스트 앞에 둠):

```
Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
...
Please select the correct answer from the options above.
```

- 주관식(53문제) : `Question: {question}`만 사용.
- `{question}` 안의 `<image 1>` 이미지 표시는 데이터 원문 그대로 유지(공식 코드와 동일).
- **출처**: Qwen 공식 평가 코드 [`QwenLM/Qwen3-VL` `evaluation/mmmu/run_mmmu.py`](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu)의 `build_mmmu_prompt()`. 이 함수는 VLMEvalKit `ImageMCQDataset.build_prompt`와 같은 형식. Qwen3-VL Technical Report(arXiv 2511.21631) Appendix B.1의 MMMU 프롬프트와 일치.
- **선택 이유**: 공식 결과와 동일한 프롬프트를 사용함으로써 격차를 프롬프트 외 요인으로 좁히기 위함. "글자만 출력" 같은 지시를 붙이지 않아 모델이 풀이를 한 뒤 대답하도록 유도 가능.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | True (vLLM sampling) |
| `temperature` | 0.7 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 3407 |

- **출처**: pinned 모델 카드 [`Qwen/Qwen3-VL-4B-Instruct` @ebb281ec](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17) README "Generation Hyperparameters"의 **VL** recipe. 같은 저장소의 `generation_config.json`과 동일한 값.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | 16384 (`max_model_len` 32768) |
| 이미지 해상도 처리 | `min_pixels = 1280·28·28`, `max_pixels = 5120·28·28` |

**선택 근거**:

- **`max_new_tokens` 16384**
  모델 카드 VL recipe의 `out_seq_length` 값. 응답 길이 측정 결과 2024 초과가 240개(26.7%), 8192 초과가 115개(12.8%), 16384 에서 85개 (9.4%)가 잘림. 입력은 최대 약 5.6 토큰 (중앙값 약 1.1천)으로 `max_model_len` 32768이 16384 출력을 제한하지 않음.

  | | 16384 (채택) | 32768 (공식 평가 코드 값) |
  |---|---|---|
  | 점수 (Qwen 파서 + 추출) | 62.11 | 62.00 |
  | 잘림 | 85 | 98 |
  | 응답 길이 16384를 넘겨 정답처리된 문제 수 | — | 1  |
  | 생성 시간 (`llm.generate()`) | 42분 | 2시간 24분 (3.4배) |

  예산을 두 배로 늘려도 점수 차이가 거의 없고 시간만 3.4배 늘어남. 16384에서 잘린 응답 대부분이 같은 문장을 되풀이하는 반복 루프로, 출력 한도를 늘려도 한도까지 반복할 뿐 정답률에 유의미한 영향을 끼치지 않았음. 따라서 16384로 확정하였음.

- **이미지 해상도**
  - **출처**: 공식 결과와 동일한 이미지 입력 조건을 맞추기 위해 Qwen 공식 MMMU 평가 코드 [`QwenLM/Qwen3-VL` `evaluation/mmmu`](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu)의 `run_mmmu.py`에 있는 `min_pixels = 1280*28*28`, `max_pixels = 5120*28*28`을 사용하였음.
  - 리사이즈 후 평가에 사용된 이미지 982장 모두 `min_pixels`(약 1.0MP)~`max_pixels`(약 4.0MP) 범위. 그중 894장이 `min_pixels` 바로 위(1.1배 이내)에 모여 있어, 대부분의 원본 이미지가 1MP보다 작아 확대된 것으로 확인됨.
  
## 4. 채점(파싱) 방식

- **사용한 파서/로직**: 공식 파서에 응답을 넘기기 전에 최종 답만 잘라내는 전처리를 붙임.

  - 전처리 `extract_final_answer()`: 자체 구현([`scripts/score.py`](../scripts/score.py)). 규칙은 [hendrycks/math](https://github.com/hendrycks/math) `last_boxed_only_string`과 [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) `extract_answer` / `extract_again`에서 차용.
  - 파서: Qwen 공식 MMMU 평가 코드의 규칙 기반 파서 `can_infer`를 수정 없이 가져옴([`QwenLM/Qwen3-VL` `evaluation/mmmu/eval_utils.py` L171-231 @96588727](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/eval_utils.py#L171-L231) → [`scripts/third_party/qwen_matching.py`](../scripts/third_party/qwen_matching.py)). 이 파서의 원 출처 [VLMEvalKit `vlmeval/utils/matching_util.py`](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/utils/matching_util.py)
- **동작 방식 요약**:
  1. **최종 답 추출** (`extract_final_answer()`): 응답에서 **가장 뒤에 나오는 표시**를 찾아 그 뒤의 답만 남김.
     - `\boxed{…}` (boxed): 마지막 `\boxed{}`의 내용
     - `answer is` / `Answer:` (phrase): 마지막 표시 뒤 그 줄의 텍스트
     - 두 표시가 모두 있으면 더 뒤에 나온 쪽을 사용. 
     - 마크다운 `**`, `$`와 `\text{}` 감싸기 제거
     - 객관식 추출 결과가 `C`, `C.`, `(C)`로 시작하면 그 글자만 넘김
     - 아무 표시를 찾지 못한 경우(표시 없음) 응답 전체를 그대로 넘김
     - 적용 결과: phrase 743, boxed 68, 표시 없음 89 (합계 900)
  2. **Qwen 공식 파서** (`can_infer`): 전처리된 응답에서 보기 글자(A,B,C,D,...)를 찾고, 없으면 보기 내용과 매칭함. 주관식은 `{A: 정답, B: "Other Answers"}` 2지선다로 보기를 만들어 판정함.
  3. **실패 시 fallback**: 공식 파서는 규칙이 실패한 경우 GPT-3.5 judge에게 넘김. 그래도 실패하면 무작위로 고름. 우리는 재현성과 API 의존성 때문에 **judge와 무작위 선택을 모두 쓰지 않고 규칙 실패를 오답으로 처리하였음**. 16384 토큰에서 잘린 응답도 같은 과정을 거치며, 최종 답이 없어 대부분 오답이 됨.(85개 중 7개는 루프 도중 문장이 우연히 정답으로 잡힘, 8절 참고).

## 5. 결과

| No. | Subject | Data Num | Acc | (참고) 잘림 |
|---|---|---|---|---|
| 1 | Accounting | 30 | 73.33 | 5 |
| 2 | Agriculture | 30 | 53.33 | 0 |
| 3 | Architecture_and_Engineering | 30 | 46.67 | 12 |
| 4 | Art | 30 | 60.00 | 3 |
| 5 | Art_Theory | 30 | 86.67 | 1 |
| 6 | Basic_Medical_Science | 30 | 73.33 | 0 |
| 7 | Biology | 30 | 50.00 | 2 |
| 8 | Chemistry | 30 | 33.33 | 1 |
| 9 | Clinical_Medicine | 30 | 60.00 | 0 |
| 10 | Computer_Science | 30 | 60.00 | 0 |
| 11 | Design | 30 | 76.67 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 30.00 | 0 |
| 13 | Economics | 30 | 76.67 | 0 |
| 14 | Electronics | 30 | 46.67 | 5 |
| 15 | Energy_and_Power | 30 | 56.67 | 12 |
| 16 | Finance | 30 | 60.00 | 3 |
| 17 | Geography | 30 | 43.33 | 2 |
| 18 | History | 30 | 70.00 | 0 |
| 19 | Literature | 30 | 83.33 | 0 |
| 20 | Manage | 30 | 66.67 | 0 |
| 21 | Marketing | 30 | 83.33 | 0 |
| 22 | Materials | 30 | 53.33 | 11 |
| 23 | Math | 30 | 56.67 | 1 |
| 24 | Mechanical_Engineering | 30 | 43.33 | 13 |
| 25 | Music | 30 | 30.00 | 11 |
| 26 | Pharmacy | 30 | 76.67 | 2 |
| 27 | Physics | 30 | 76.67 | 0 |
| 28 | Psychology | 30 | 80.00 | 0 |
| 29 | Public_Health | 30 | 90.00 | 1 |
| 30 | Sociology | 30 | 66.67 | 0 |
| | **Overall (macro avg)** | **900** | **62.11** | 85 |

계산식: `Overall = (1/30) · Σ_s acc_s`, `acc_s = 과목 s의 정답 수 / 30 × 100`. 과목당 문제 수가 동일함으로 micro 정확도(559/900 = 62.11%)와 동일함.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 62.11 |
| 차이 (Δ) | −5.29 |


## 7. 격차 분석

공식 값과의 격차 5.29점 중 약 3.6점은 채점 방식 차이, 나머지는 실행 간 변동으로 설명됨. 

**① judge 생략 (+1.7%)**: 응답이 끝까지 생성되었지만 규칙 파싱에 실패한 38개의 문제들의 추출 결과를 확인해본 결과, 표기 차이 8개(`\dfrac{1}{64}` vs `1/64`, `1,000` vs `1000`), 추출 실패 2개, 반올림 차이 4개(13.0 vs 12.97, 24.3 vs 24.32) 등 llm judge를 사용하였더만 정답 처리되었을 가능성이 있는 문제 약 15개 확인됨. 

**② 무작위 선택 생략 (+1.9점)**: 공식 채점 방식에서 llm judge가 답을 못 찾는 경우 보기를 무작위로 고르지만, 우리는 모두 오답 처리함. 루프로 잘려 답이 없는 58개에 무작위 선택을 적용하면 기대 정답은 약 17개.

**③ 샘플링 변동 (±1.45점 그 이상)**: 첫 실행과 결과와 재실행 결과를 **부분적으로**  비교하였을 때, 첫 실행에서 잘린 39개가 재실행에서는 끝까지 답해 22개를 맞혔지만, 다른 32개가 새로 루프에 빠짐. 그 결과 점수가 62.11 → 63.56으로 달라짐.


## 8. 기타 특이사항 / 한계 (Optional)

- **잘린 응답의 우연한 정답**: 잘린 85개 중 7개는 루프 도중 문장이 우연히 정답으로 잡혀 점수에 합산됨. 모델이 답을 낸 것이 아님. 잘린 응답을 모두 오답으로 처리하면 61.33점(552/900), 보고 점수는 파이프라인이 그대로 출력하는 62.11로 두고 잘림 수를 함께 보고하였음.

- **참고 실행 3회 비교** (보고 점수는 1차 실행 결과)

  | | 1차 실행 (보고) | 2차 실행 (동일한 설정으로 재실행) | 3차 실행 (`max-new-tokens`=32768) |
  |---|---|---|---|
  | 점수 (Qwen 파서 + 추출) | **62.11** | 63.56 | 62.00 |
  | 잘림 | 85 | 78 | 98 |
  | 끝까지 생성된 응답의 정확도 | 67.73% | 68.37% | 68.20% |
  | 첫 실행과 응답이 완전히 같은 문제 | — | 145 / 900 | 132 / 900 |
  | 첫 실행과 정답 여부가 같은 문제 | — | 807 / 900 | 813 / 900 |

- **다음에 시도해 보고 싶은 것 (루프 줄이기)**

  반복 실행은 3회뿐이었지만 실행마다 샘플링이 달라지고 루프가 다른 문제로 옮겨 가는 것을 확인하였음. 루프를 줄이면 풀이 능력과 별개로 점수를 높일 수 있을 것으로 예상.
  - **Self-consistency** ([Wang et al., 2023](https://arxiv.org/abs/2203.11171)): 같은 문제를 여러 번 샘플링해 각 응답의 최종 답을 뽑고, 가장 많이 나온 답을 채택. 루프에 걸린 샘플은 투표에서 제외함.
  - **Budget forcing** ([Muennighoff et al., 2025, s1](https://arxiv.org/abs/2501.19393)): 토큰 한도에 닿거나 루프가 감지되면 생성을 끊고 "Final Answer:"를 붙여 답만 생성하도록 함.

