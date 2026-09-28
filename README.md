# MMDL

멀티모달 딥러닝 수업 팀 repo입니다. 과제 1에서는 **Qwen3-VL-4B-Instruct를 MMMU validation(30과목 × 30문제 = 900문제)으로 평가하는 재현 가능한 파이프라인**을 만들었습니다. 이후 이 모델을 fine-tuning한 체크포인트도 같은 파이프라인으로 다시 평가해 비교합니다.

- **팀명**: 1조
- **팀원**: 김유진, 류승희, 송지훈

## 결과 요약

| | Overall (MMMU val, macro avg) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| **우리 재현 결과** | **62.11** (559/900) |

- 제출 보고서: [reports/mmmu_baseline.md](reports/mmmu_baseline.md) (환경, 프롬프트, 생성 설정, 채점 방식, 과목별 결과, 격차 분석)
- 평가 결과 분석: [reports/mmmu_eval_analysis.md](reports/mmmu_eval_analysis.md) (파싱 검증, 과목별 잘림·루프, 오답 341개 원인 분류)

## 빠른 시작

### 1. 환경

- GPU 필요 (사용 환경: A100-SXM4-80GB 1장. 24GB GPU)
- **헤더(`Python.h`)가 포함된 Python 3.12**가 필요합니다. vLLM 엔진이 뜰 때 Triton이 C 확장을 컴파일하기 때문입니다. 시스템 Python에 헤더가 없으면 uv가 관리하는 Python을 씁니다.

```bash
uv venv --seed --python 3.12 .venv
source .venv/bin/activate
pip install -r requirements.lock.txt   # 실제 실행 환경 고정 (최소 버전만 필요하면 requirements.txt)
```

### 2. 평가 실행

```bash
bash scripts/run_mmmu_eval.sh \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --data_root <MMMU HF datasets cache> \
  --output_dir outputs/qwen3vl4b_mmmu_val
```

- `--model_path`: HF repo id 또는 로컬 체크포인트 디렉터리
- `--data_root`: MMMU HF datasets 캐시 경로 (생략하면 HF 기본 캐시)
- 일부 과목만 빠르게 확인하려면 `--subjects Math` 처럼 추가합니다.

### 3. 재채점 (CPU, 추론 x)

```bash
python scripts/score.py --pred outputs/qwen3vl4b_mmmu_val/predictions.jsonl \
  --parser qwen --extract final --output_dir outputs/qwen3vl4b_mmmu_val
python scripts/test_score.py   # 프롬프트 구성과 두 채점기의 CPU 스모크 테스트 (numpy만 필요)
```

## 고정 설정

자세한 출처와 선택 근거는 [보고서](reports/mmmu_baseline.md) 참고.

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3-VL-4B-Instruct` @`ebb281ec70b05090aa6165b016eac8ec08e71b17`, bf16 |
| 데이터 | HF `MMMU/MMMU` @`98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, `validation`, 30과목 전부 |
| 프롬프트 | Qwen 공식 `evaluation/mmmu/run_mmmu.py`의 `build_mmmu_prompt()`와 동일, 이미지는 텍스트 앞 |
| 샘플링 | 모델 카드 VL recipe: temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5, seed 3407 |
| 생성 예산 | `max_new_tokens` 16384, `max_model_len` 32768 |
| 이미지 | `min_pixels` = 1280·28·28, `max_pixels` = 5120·28·28 (공식 코드 값) |
| 채점 (보고 점수) | 최종 답 추출 → Qwen 공식 규칙 파서. judge·무작위 선택 없이 실패는 오답 |
| 종합 점수 | 30개 과목 정확도의 단순 평균 (macro) |

## 폴더 구조

```
scripts/
  run_mmmu_eval.sh      명령 하나로 추론 + 채점
  infer.py              vLLM 추론 → predictions.jsonl, predictions_meta.json
  score.py              채점 (--parser qwen|mmmu, --extract final|none)
  test_score.py         CPU 스모크 테스트
  third_party/          Qwen·MMMU 공식 파서 원본 복사본 (수정 금지)
outputs/
  qwen3vl4b_mmmu_val/         보고 점수 실행 (1차 실행)
  qwen3vl4b_mmmu_val_rerun/   같은 설정 재실행 (2차 실행)
  qwen3vl4b_mmmu_val_32k/     max_new_tokens 32768 실행 (생성 예산 비교용)
reports/                보고서와 분석 문서
docs/                   과제 안내와 제출 템플릿
```

## 출력 파일

| 파일 | 내용 |
|---|---|
| `predictions.jsonl` | 문제별 원본 응답, 정답, `finish_reason`, 출력 토큰 수, 이미지 크기 |
| `predictions_meta.json` | 실행 인자, 라이브러리 버전, GPU, 소요 시간, peak VRAM, 잘림 수(`num_truncated`) |
| `scores_qwen.{csv,json,md}` | **보고 점수** (Qwen 파서 + 최종 답 추출) |
| `scores_mmmu.*` | 비교용: MMMU 공식 파서 + 추출 |
| `scores_{qwen,mmmu}_raw.*` | 비교용: 추출 없이 응답 전체를 파서에 넣은 원본 방식 |

## fine-tuning 후 재평가

```bash
bash scripts/run_mmmu_eval.sh --model_path <ckpt dir> --model_revision '' \
  --data_root <MMMU HF datasets cache> --output_dir outputs/<name>
```

- 프롬프트, 샘플링, 추출 규칙, 파서는 **바꾸지 않고** `scores_qwen.md`끼리 비교합니다.
- 같은 설정에서도 한 번 실행한 점수가 약 1.5점 흔들리므로, 차이가 작으면 문제별 짝 비교(McNemar)로 우연인지 확인합니다.
- 잘림 수(`predictions_meta.json`의 `num_truncated`)도 함께 비교합니다. 반복 루프로 인한 잘림이 줄면 풀이 능력과 별개로 점수가 오를 수 있습니다.

## 주의 사항

- 모델 가중치는 커밋하지 않습니다. HF 캐시에서 revision을 고정해 내려받습니다.
- `scripts/third_party/`의 공식 파서는 수정하지 않습니다. 필요한 조정은 `score.py`에서 합니다.
