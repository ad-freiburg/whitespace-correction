#!/bin/bash
set -e

base_dir=$(dirname "$0")
if [[ $1 == "" ]]; then
  split=org
else
  split=$1
fi

echo "Base directory: ${base_dir}"

benchmarks=${base_dir}/${split}_benchmarks
results=${base_dir}/${split}_results

EXPERIMENTS=$(echo ${base_dir}/../../experiments/*)

EXP_REGEX=${EXP_REGEX:-"*"}

BENCHMARK_REGEX=${BENCHMARK_REGEX:-"*"}

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

  test_benchmarks=${benchmarks}/${BENCHMARK:-*}/test/corrupt.txt

  for benchmark in $test_benchmarks
  do
    if [[ $benchmark != $BENCHMARK_REGEX ]]; then
      continue
    fi
    trt -e $exp \
    -f $benchmark \
    -o ${results}/$(dirname $(realpath $benchmark --relative-to $benchmarks))/${model_name}.txt
  done
done
