#!/bin/bash
set -e

BASE_DIR=$(dirname "$0")

echo "Base directory: $BASE_DIR"

python $BASE_DIR/../evaluate.py --groundtruths $BASE_DIR/cleaned_benchmarks/*/test/correct.txt \
  --predictions $BASE_DIR/cleaned_results/*/test/*.txt --save-markdown-dir $BASE_DIR/evaluate_tables