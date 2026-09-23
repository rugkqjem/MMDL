#!/bin/bash


# 2. 결과 저장 디렉토리 생성
OUTPUT_DIR="./results"
mkdir -p $OUTPUT_DIR

# 3. eval_mmmu.py 실행 ($1 인자로 전달받은 작업 디렉터리 기준)
echo "Starting MMMU Evaluation..."

# 현재 run.sh가 위치한 디렉토리 경로 가져오기
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python "$SCRIPT_DIR/eval_mmmu.py" \
    --model_id "Qwen/Qwen3-VL-4B-Instruct" \
    --revision_id "ebb281ec70b05090aa6165b016eac8ec08e71b17" \
    --max-new-tokens 128 \
    --temperature 1 \
    --top-p 1.0 \
    --top-k 40 \
    --repetition-penalty 1.0\
    --presence-penalty 2.0\
    --output_dir "$SCRIPT_DIR/results"

echo "Evaluation completed! Report saved in $SCRIPT_DIR/results/mmmu_evaluation_report.csv"