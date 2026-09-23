# HANDOFF — assignment1-vllm-eval (2026-09-24)

코드는 노트북에서 작성했고, 노트북에서는 `python scripts/test_score.py`만 통과를 확인함. **`infer.py`의 vLLM 부분은 아직 한 번도 실행해 보지 않았음.** 배경과 출처는 `AGENTS.md` 참고.

## GPU(A100 80GB)에서 할 일

1. 환경 구성
   ```bash
   git clone -b assignment1-vllm-eval <repo-url> && cd MMDL
   python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```
2. Math 한 과목으로 테스트 (16384로 한 번만 돌리면 2048 기준 잘림 수도 같이 계산 가능)
   ```bash
   bash scripts/run_mmmu_eval.sh --data_root ~/hf_data --output_dir outputs/smoke_math --subjects Math --max_new_tokens 16384
   python -c "import json;r=[json.loads(l) for l in open('outputs/smoke_math/predictions.jsonl')];t=sorted(x['num_output_tokens'] for x in r);print('tokens',t);print('>2048:',sum(x>2048 for x in t),'/',len(t),'| hit 16384:',sum(x['finish_reason']=='length' for x in r))"
   cat outputs/smoke_math/predictions_meta.json
   ```
   - vLLM 버전에 따라 `LLM(revision=, tokenizer_revision=)`이나 `mm_processor_kwargs` 쪽에서 에러가 날 수 있음 → 에러 로그를 확인하고 수정
   - `image_sizes`를 보고 `min_pixels`/`max_pixels`(28*28 기준 상수)가 실제로 어떻게 적용됐는지 기록
3. `max_new_tokens` 결정: 16384에서 잘림이 거의 없고 시간이 감당할 만하면 `infer.py` 기본값을 16384(모델 카드 VL recipe의 `out_seq_length`)로 변경. 아니면 2048을 유지하고 잘림 비율을 근거로 보고서에 기록
4. 전체 900문제 실행 (`--subjects` 없이) 후 의존성 고정
   ```bash
   bash scripts/run_mmmu_eval.sh --data_root ~/hf_data --output_dir outputs/qwen3vl4b_mmmu_val
   pip freeze > requirements.lock.txt
   ```
5. `outputs/qwen3vl4b_mmmu_val/`(predictions, meta, scores_*)와 `requirements.lock.txt`를 커밋

## 그다음 (노트북에서 가능)

- `reports/mmmu_baseline.md` 작성 (`docs/SUBMISSION_TEMPLATE.md` 형식). 결과 표는 `scores_qwen.md`를 붙여 넣으면 됨
- 격차 분석 근거: 잘림 비율, qwen 파서의 `fail`/`rule_Z` 비율, qwen과 mmmu 파서 점수 차이, 기존 `assignment1-eval` 브랜치 결과(53.89, Text recipe + "letter만 출력" + 128 tokens)와 비교
- 시간이 되면 `--seed 42`로 한 번 더 돌려서 seed만 바꿨을 때 점수가 얼마나 흔들리는지 측정
- 마감: 2026-09-28 23:59
