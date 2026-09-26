# HANDOFF — assignment1-vllm-eval (2026-09-26)

GPU(A100 80GB)에서 900문제 전체 실행과 채점까지 끝남. 배경과 출처는 `AGENTS.md`, 결과·설정·격차 분석·회의 안건은 **`BASELINE_NOTES.md`** 참고.

## 완료
- 환경: Python 3.12.14 venv (헤더 포함 — Triton 컴파일에 `Python.h` 필요), 버전 고정은 `requirements.lock.txt`
- 09-23 실행은 `max_new_tokens` 16384 (모델 카드 VL recipe `out_seq_length`), `max_model_len` 32768. Math 30문제 테스트에서 50%가 2048 초과
- 전체 실행: `outputs/qwen3vl4b_mmmu_val/` (생성 42분, 잘림 85/900)
- 09-25: 공식 MMMU 평가 코드(`run_mmmu.py`, `infer_instruct.sh`)가 `--max-new-tokens 32768 --max-model-len 128000`인 것을 확인하고 이 값으로 재실행 → `outputs/qwen3vl4b_mmmu_val_32k/`
  - 점수 62.00 (16384는 62.11), 잘림 98, 순수 추론 2시간 24분 (3.4배). 잘림은 대부분 반복 루프라 예산을 늘려도 해결 안 됨
  - 같은 seed여도 문제의 약 10%가 정답 여부가 바뀜 (샘플링 노이즈)
  - → 기본값은 **16384 / 32768 유지**
- 09-26: 첫 실행과 같은 설정으로 재실행 → `outputs/qwen3vl4b_mmmu_val_rerun/` (63.56, 잘림 78). 응답이 145/900만 같아서, 32768과의 차이도 설정 변경이 아닌 실행 간 변동으로 확인
- **보고 점수는 첫 실행 한 번의 결과(62.11)와 그 응답만 사용** (여러 실행 평균 안 씀). `_32k`, `_rerun`은 참고 자료
- 이미지 해상도 처리를 공식 코드·공식 TSV와 대조: 해상도는 같음 (JPEG/PNG, 투명 배경 12장만 차이)
- 채점: 파서 전에 최종 답 추출 단계 추가 (공개 선례가 있는 규칙만: 마지막 `\boxed{}`, `answer is` / `Answer:`)
  - 주 지표 Qwen 파서 + 추출 **62.11**, MMMU 파서 + 추출 62.22, 추출 없이 원본 방식 32.67 / 50.78
- 팀 회의용 정리 노트 `BASELINE_NOTES.md` 작성

## 남은 일
- 팀 회의: `BASELINE_NOTES.md`의 "회의에서 정할 것" (주 지표, judge, 잘린 응답 처리, 분담·팀명)
- 회의 결과를 반영해 제출 보고서 `reports/mmmu_baseline.md` 작성 (`docs/SUBMISSION_TEMPLATE.md` 형식)
- `git push` (git 설정/토큰은 사용자가 직접)
- 마감: 2026-09-28 23:59

## fine-tuning 후 재평가
```bash
bash scripts/run_mmmu_eval.sh --model_path <ckpt dir> --model_revision '' --data_root <MMMU HF cache> --output_dir outputs/<name>
```
프롬프트, 샘플링, 추출 규칙, 파서를 바꾸지 말고 한 번 실행한 `scores_qwen.md`끼리 비교. 잘림 수(`predictions_meta.json`의 `num_truncated`)도 함께 비교할 것. 같은 설정에서도 실행마다 약 1.5점 흔들릴 수 있으니, 차이가 작으면 문제별 짝 비교(McNemar)로 확인.
