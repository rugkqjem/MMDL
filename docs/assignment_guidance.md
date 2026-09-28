# Assignment: Qwen3-VL-4B MMMU Baseline Evaluation

## 0. 개요

이 과제의 산출물은 "Qwen3-VL-4B-Instruct를 MMMU validation split으로 평가한 표"입니다.
하지만 **진짜 목적은 표가 아니라 평가 파이프라인 그 자체**입니다.

이 수업에서 여러분은 이 backbone(Qwen3-VL-4B)을 fine-tune하게 됩니다. 그때 "fine-tuning이 실제로
도움이 됐는가"를 말하려면, **지금 만든 것과 정확히 같은 방식으로 다시 평가**해서 두 숫자를 비교할 수 있어야
합니다. 두 실행 사이에 프롬프트가 살짝 다르거나, 채점 방식이 바뀌거나, 생성 파라미터가 달라지면 그 비교는
의미를 잃습니다.

따라서 이 과제는 "점수를 최대한 official 수치(67.4)에 가깝게 맞추는 것"이 목표가 아닙니다.
**여러분의 점수는 공식 수치보다 낮아도, 높아도 됩니다.** 대신 다음을 요구합니다:

1. 지금 만든 파이프라인을 **하나의 커맨드로 재현 가능**하게 만들 것 (나중에 체크포인트만 바꿔서 다시 돌릴 수 있게)
2. 파이프라인의 모든 선택(프롬프트, 생성 설정, 채점 방식)을 **명시적으로 문서화**할 것
3. 공식 수치와의 차이를 관찰하고, **왜 차이가 나는지 스스로 진단**할 것

## 1. 과제 스펙

### 1.1 모델 (고정 — 팀마다 달라지면 채점/비교가 불가능해짐)

| 항목 | 값 |
|---|---|
| HuggingFace repo | **`Qwen/Qwen3-VL-4B-Instruct`** (공식 `Qwen` org 배포본만 인정 — 커뮤니티 재업로드/양자화(GGUF, AWQ 등) 버전 금지) |
| Revision (commit sha) | **`ebb281ec70b05090aa6165b016eac8ec08e71b17`** — 반드시 이 revision을 pin해서 받으십시오 (`revision=` 인자 또는 동일 커밋의 snapshot). 모델이 이후 업데이트되어도 팀마다 다른 버전을 쓰게 되는 걸 방지하기 위함입니다. |
| dtype | bf16 (`torch_dtype="auto"` 또는 `torch.bfloat16` — 리포지토리 기본값 그대로, 별도 양자화/캐스팅 금지) |
| 다운로드 방법 | `huggingface_hub`/`transformers`의 표준 캐시 방식 권장 (`from_pretrained(..., revision=...)`). 팀 repo에 가중치 자체를 커밋하지 말 것. |

### 1.2 데이터셋 (고정)

| 항목 | 값 |
|---|---|
| HuggingFace repo | **`MMMU/MMMU`** (dataset) |
| Revision (commit sha) | **`98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`** — 모델과 동일한 이유로 pin. |
| Split | **validation** |
| 과목(config) | 아래 30개 **전부** 각각 별도 config로 로드해야 합니다 (`load_dataset("MMMU/MMMU", "<과목명>", split="validation", revision=...)`) — MMMU는 하나의 config로 전체 과목이 안 나옵니다. |

<details>
<summary>30개 과목 config 이름 (펼치기)</summary>

```
Accounting, Agriculture, Architecture_and_Engineering, Art, Art_Theory,
Basic_Medical_Science, Biology, Chemistry, Clinical_Medicine, Computer_Science,
Design, Diagnostics_and_Laboratory_Medicine, Economics, Electronics,
Energy_and_Power, Finance, Geography, History, Literature, Manage,
Marketing, Materials, Math, Mechanical_Engineering, Music, Pharmacy,
Physics, Psychology, Public_Health, Sociology
```
</details>

데이터를 임의로 필터링/서브샘플링하지 마십시오 (과목당 30문제, 총 900문제 그대로).

### 1.3 하드웨어 / 실행 방식 (자유 — 단, 문서화 필수)

| 항목 | 값 |
|---|---|
| 참고 하한 사양 | 실습실 **RTX 4090(24GB)** 기준으로도 이 파이프라인이 돌아갈 수 있어야 함 — 다만 이건 "이 정도 사양이면 충분하다"는 하한 참고치일 뿐, 모든 팀이 실제로 24GB 이하에서 실행해야 한다는 강제 규정은 아닙니다. 본인 인프라에 맞게 설계하고 그 근거를 문서화하면 됩니다. |
| 추론 백엔드 | 자유 (`transformers.generate()`, `vLLM` 등 — 단, 선택 이유를 문서화) |
| **Sampling recipe** (`temperature`/`top_p`/`top_k`/`repetition_penalty`/`presence_penalty`/`seed`) | **임의로 정하지 말 것.** 모델 제공사(Qwen)가 이 모델의 MMMU 등 벤치마크 재현을 위해 공식 문서화한 generation recipe가 존재합니다. **이를 직접 찾아서 사용하고, 어디서 찾았는지 출처를 명시**하십시오. 공식 recipe를 못 찾았다고 주장하거나, 찾았지만 다른 값(예: greedy)을 쓰기로 했다면 **그 근거를 명확히 해야** 합니다 — "greedy가 안전할 것 같아서" 처럼 근거가 약한 경우는 credit을 받지 못합니다. |
| **생성 예산 / 이미지 해상도** (`max_new_tokens`, `min_pixels`/`max_pixels` 등) | **자유 - GPU 인프라 제약과 직접 연결된 엔지니어링 판단**이므로 여러분이 직접 정하고 근거(속도/VRAM/정답 잘림 등 trade-off)를 문서화하십시오. "정답 값"은 없습니다. |

## 2. 요구 산출물

### 2.1 과목별 + 종합 점수 표

30개 과목 각각의 정확도(accuracy)와, **종합 점수**를 표로 제출합니다.

> **종합 점수 = 30개 과목 정확도의 단순 평균 (macro average)**
> (MMMU val은 과목당 문제 수가 정확히 30개로 균등하기 때문에, 이는 "전체 900문제 중 맞힌 개수의 비율"과
> 수학적으로 동일합니다. 다만 과목별 문제 수가 다른 데이터셋/split에서는 두 방식이 달라지므로,
> **어떤 방식으로 종합 점수를 계산했는지 명시**하십시오.)

### 2.2 파이프라인 문서화

아래 세 가지를 **정확히, 재현 가능하게** 기술하십시오. "GPT를 써서 물어봤다" 수준이 아니라, 다른 사람이
여러분의 문서만 보고 동일한 파이프라인을 그대로 재구성할 수 있는 수준이어야 합니다.

- **(a) 프롬프트**: 모델에 실제로 들어간 프롬프트 전문(template). 어디서 가져왔는지(직접 설계 / 특정
  evaluation toolkit에서 차용 / 모델 제공사가 공개한 예시 등) 출처를 밝히십시오.
- **(b) 생성 설정**: `max_new_tokens`, sampling 여부(`temperature`, `top_p`, `top_k` 등), 이미지 해상도
  처리(`min_pixels`/`max_pixels` 등), 그리고 이 값들을 **왜 그렇게 정했는지**.
- **(c) 채점(파싱) 방식**: 모델의 자유 텍스트 응답에서 최종 선택지(A/B/C/D...)를 뽑아내는 방법. 직접
  구현했다면 그 로직을, 기존의 공개된 도구(오픈소스코드)를 썼다면 어느 것을 썼는지 밝히십시오.

### 2.3 공식 수치와의 비교 및 격차 분석

- MMMU **공식 수치(Qwen3-VL Technical Report 기준)**는 종합 점수 기준 67.4점 입니다. 직접 측정한 점수와 종합 점수를 나란히 제시.
- 차이가 있다면, **왜 차이가 나는지 1000자 이내로 진단**하십시오 (짧고 근거가 명확할수록 좋습니다 — 분량이
  credit을 늘려주지 않습니다). 2.2에서 문서화한 선택들(프롬프트, 생성 설정, 채점 방식) 중 무엇이 원인일 수 있는지, **근거를 들어** 추론하십시오. 가능하다면 그 원인을 뒷받침하는 간단한 추가 실험(예: 특정 과목의 응답 샘플 몇 개를 직접 열어보고 패턴 확인)을 곁들이면 좋습니다.
- 완벽하게 원인을 규명하지 못해도 괜찮습니다 — **추론의 질(evidence-based reasoning)**이 평가 대상입니다.

## 3. 채점 (100점)

| 항목 | 배점 |
|---|---|
| A. 환경/재현성 + 제출 형식 (한 커맨드로 재실행 가능, 의존성/경로 명시, 템플릿 준수) | 20 |
| B. 실행 완전성 + 결과 표 (900문제 전체 실행, 30과목+종합 표, 계산식 명시 및 산술 정합) | 25 |
| C. 프롬프트 문서화 (전문 + 출처) | 10 |
| D. 생성 설정 문서화 (파라미터 + 근거) | 10 |
| E. 채점/파싱 방식 문서화 | 10 |
| F. 공식 수치 비교 + 격차 분석의 깊이/타당성 | 25 |

세부 채점 기준은 `SUBMISSION_TEMPLATE.md`의 항목 구조를 따릅니다.

## 4. 제출 방법

- 팀 GitHub repo 내 **`reports/mmmu_baseline.md`** 에 작성.
- `SUBMISSION_TEMPLATE.md`를 복사해서 사용하십시오.
- 평가를 재현하는 데 필요한 스크립트/커맨드는 같은 repo 안에 포함.
- 마감: 2026. 9. 28. 월요일. 23:59 까지

## 5. 유의사항

- 순정 `transformers.generate()`만으로는 900문제 전체 평가가 매우 오래 걸릴 수 있습니다. 배치/서빙
  프레임워크(vLLM 등) 활용은 자유이며 오히려 권장됩니다. 단, 어떤 선택을 했든 **왜** 그렇게 했는지
  1.3(추론 백엔드) — 제출 시엔 `SUBMISSION_TEMPLATE.md`의 "1. 환경 / 재현성" 섹션의 "추론 백엔드"
  항목 — 에 설명하십시오. 생성 길이 제한(`max_new_tokens`) 관련 근거는 별도로 2.2(b)/템플릿 "3. 생성
  설정" 섹션에 적으면 됩니다.
- 이 과제의 재현성 기준(문서화 수준)은 fine-tuning 이후 재평가 때 그대로 재사용됩니다. 기준을 잘 정해야 합니다.
