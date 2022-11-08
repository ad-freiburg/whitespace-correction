#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --nodes=1
#SBATCH --mincpus=8
#SBATCH --mem=64G
#SBATCH --job-name=preprocessing
#SBATCH --mail-user=swalter@cs.uni-freiburg.de
#SBATCH --mail-type=END,FAIL
#SBATCH --time=12:00:00

workspace=/work/dlclarge2/swalter-whitespace_correction

python $workspace/whitespace-correction/src/whitespace_correction/preprocess_data.py \
 --config ${CONFIG?"CONFIG env variable not found"}
