#!/bin/bash
set -e

BASE_DIR=$(dirname "$0")

echo "Base directory: $BASE_DIR"

CLEANED_BENCHMARKS="$BASE_DIR/cleaned_benchmarks"
CLEANED_RESULTS="$BASE_DIR/cleaned_results"

EXPERIMENTS=$(echo $BASE_DIR/../../experiments/*)

EXP_REGEX=${EXP_REGEX:-"*"}

echo "Enter device string to use (e.g. 'cpu' or 'cuda'):"

read device_string

for exp in $EXPERIMENTS
do
  experiment_name=$(basename "$exp")
  if [[ $experiment_name != $EXP_REGEX ]]; then
    echo "Skipping $experiment_name because it does not match the regex $EXP_REGEX"
    continue
  fi

  echo "Enter model name for experiment $experiment_name (or type 'skip' to skip):"

  read model_name
  if [[ "$model_name" == "skip" ]]; then
    echo "Skipping experiment $experiment_name"
    continue
  fi

  benchmarks=$CLEANED_BENCHMARKS/${BENCHMARK:-*}/test/corrupt.txt

  python $BASE_DIR/../run.py --experiment $exp -bs ${BATCH_SIZE:-16} -sb -d $device_string -sp \
    -tr --benchmarks $benchmarks --output-dir $CLEANED_RESULTS \
    --save-markdown-dir $BASE_DIR/runtime_tables --model-name $model_name

done
