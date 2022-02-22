#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --nodes=1
#SBATCH --mincpus=32
#SBATCH --mem=256G
#SBATCH --job-name=preprocessing_tokenization_repair
#SBATCH --mail-user=sebastian.walter98@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=12:00:00

export WORKSPACE_DIR=/work/dlclarge1/swalter-tokenization_repair/tokenization_repair

export DATA_INPUT_DIR=$WORKSPACE_DIR/data/cleaned
export DATA_OUTPUT_DIR=$TMPDIR/data/preprocessed

export TOKENIZER_DIR=$WORKSPACE_DIR/tokenizers
export MISSPELLINGS_DIR=$WORKSPACE_DIR/misspellings
export DICTIONARIES_DIR=$WORKSPACE_DIR/dictionaries

echo "workspace dir: $WORKSPACE_DIR"
echo "data input dir: $DATA_INPUT_DIR"
echo "data output dir: $DATA_OUTPUT_DIR"

python -m trt.preprocess_data --config ${CONFIG?"CONFIG env variable not found"}

rsync -r -ah --progress $DATA_OUTPUT_DIR/ $WORKSPACE_DIR/data/preprocessed
