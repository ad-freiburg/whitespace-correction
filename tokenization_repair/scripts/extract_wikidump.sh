#!/bin/bash
#SBATCH --partition=single
#SBATCH --job-name=spelling_correction_extract_wikidump
#SBATCH --mail-user=sebastian.walter98@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=4:00:00
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --output=bwunicluster/logs/extract_wikidump_stdout_logs.txt
#SBATCH --error=bwunicluster/logs/extract_wikidump_stderr_logs.txt

export HOME_DIR=$HOME/sebastian-walter
export WORK_DIR=$HOME_DIR/spelling_correction

export DATA_DIR=$WORK_DIR/data

python -m wikiextractor.WikiExtractor $DATA_DIR/wikidump_20201020/enwiki-20201020-pages-articles-multistream.xml.bz2 -o \
  $DATA_DIR/wikidump_20201020/extracted