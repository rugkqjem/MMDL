# HANDOFF — assignment1-vllm-eval (2026-09-23)

GPU(A100 80GB)에서 900문제 전체 실행과 채점까지 끝남. 배경과 출처는 `AGENTS.md`, 결과·설정·격차 분석·회의 안건은 **`BASELINE_NOTES.md`** 참고.

## 완료
- 환경: Python 3.12.14 venv (헤더 포함 — Triton 컴파일에 `Python.h` 필요), 버전 고정은 `requirements.lock.txt`
- `max_new_tokens` 기본값 16384 (모델 카드 VL recipe `out_seq_length`). Math 30문제 테스트에서 50%가 2048 초과
- 전체 실행: `outputs/qwen3vl4b_mmmu_val/` (생성 42분, 잘림 85/900)
- 채점: 파서 전에 최종 답 추출 단계 추가 (공개 선례가 있는 규칙만: 마지막 `\boxed{}`, `answer is` / `Answer:`)
  - 주 지표 Qwen 파서 + 추출 **62.11**, MMMU 파서 + 추출 62.22, 추출 없이 원본 방식 32.67 / 50.78
- 팀 회의용 정리 노트 `BASELINE_NOTES.md` 작성

## 남은 일
- 팀 회의: `BASELINE_NOTES.md`의 "회의에서 정할 것" (주 지표, `max_new_tokens`, judge, 반복 실행, 분담·팀명)
- 회의 결과를 반영해 제출 보고서 `reports/mmmu_baseline.md` 작성 (`docs/SUBMISSION_TEMPLATE.md` 형식)
- `git push` (git 설정/토큰은 사용자가 직접)
- seed를 바꾼 반복 실행은 아직 하지 않음
- 마감: 2026-09-28 23:59

## fine-tuning 후 재평가
```bash
bash scripts/run_mmmu_eval.sh --model_path <ckpt dir> --model_revision '' --data_root <MMMU HF cache> --output_dir outputs/<name>
```
프롬프트, 샘플링, 추출 규칙, 파서를 바꾸지 말고 `scores_qwen.md`끼리 비교. 잘림 수(`predictions_meta.json`의 `num_truncated`)도 함께 비교할 것.
