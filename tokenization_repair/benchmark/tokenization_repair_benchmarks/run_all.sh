#!/bin/bash
set -e

base_dir=$(dirname "$0")
if [[ $1 == "" ]]; then
  split=cleaned
else
  split=$1
fi

echo "Base directory: ${base_dir}"

benchmarks=${base_dir}/${split}_benchmarks
results=${base_dir}/${split}_results

EXPERIMENTS=$(echo ${base_dir}/../../experiments/*)

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

  test_benchmarks=${benchmarks}/${BENCHMARK:-*}/test/corrupt.txt

#  python ${base_dir}/../run.py --experiment $exp -bs ${BATCH_SIZE:-16} -sb -d ${device_string} -sp \
#    -tr --benchmarks ${test_benchmarks} --output-dir ${results} \
#    --save-markdown-dir ${base_dir}/${split}_runtime_tables --model-name ${model_name}
  for benchmark in $test_benchmarks
  do
    trt -e $exp \
    -f $benchmark \
    -o ${results}/$(dirname $(realpath $benchmark --relative-to $benchmarks))/${model_name}.txt
  done
done
