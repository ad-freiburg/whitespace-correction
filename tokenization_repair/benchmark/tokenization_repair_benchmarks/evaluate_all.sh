#!/bin/bash
set -e

base_dir=$(dirname "$0")
if [[ $1 == "" ]]; then
  split=cleaned
else
  split=$1
fi

echo "Base directory: $base_dir"

python ${base_dir}/../evaluate.py --groundtruths ${base_dir}/${split}_benchmarks/*/test/correct.txt \
  --predictions ${base_dir}/${split}_results/*/test/*.txt --save-markdown-dir ${base_dir}/${split}_evaluate_tables