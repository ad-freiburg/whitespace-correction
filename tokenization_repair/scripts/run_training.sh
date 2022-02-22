#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=training_tokenization_repair
#SBATCH --mail-user=sebastian.walter98@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00

export WORKSPACE_DIR=/work/dlclarge1/swalter-tokenization_repair/tokenization_repair

export EXPERIMENT_DIR=$WORKSPACE_DIR/experiments
export TOKENIZER_DIR=$WORKSPACE_DIR/tokenizers

rsync -ah --progress $LMDB_PATH $TMPDIR/lmdb
export LMDB_PATH=$TMPDIR/lmdb

echo "work dir: $WORKSPACE_DIR"
echo "experiment dir: $EXPERIMENT_DIR"
echo "lmdb: $LMDB_PATH"

random_port=$(python -c "import random; print(random.randrange(10000, 60000))")
export TORCH_DIST_PORT=$random_port

python -W ignore -m trt.train --config ${CONFIG?"CONFIG env variable not found"}
