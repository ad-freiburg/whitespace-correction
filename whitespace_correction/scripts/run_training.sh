#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=4
#SBATCH --job-name=training
#SBATCH --output=train_%x_%j.slurm
#SBATCH --mail-user=swalter@cs.uni-freiburg.de
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00

force_local=${FORCE_LOCAL:-false}
if [[ -n $SLURM_JOB_ID ]]; then
  script_dir=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
  is_local=false
else
  script_dir=$(realpath $0)
  is_local=true
fi
script_dir=$(dirname $script_dir)
workspace=$(realpath $script_dir/../..)
code=$(realpath $workspace/src/whitespace_correction)
cd $workspace
echo "Script is located at $script_dir, workspace is $workspace, code is at $code"

if [[ $is_local == true || $force_local == true ]]; then
  echo "Running locally (force_local=$force_local)"
  master_addr="localhost"
  master_port=$(python3 -c "import random; print(random.Random().randint(10000, 60000))")
  world_size=$(python3 -c "import torch; print(torch.cuda.device_count())")
else
  master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  # set the master port to a random port on the slurm cluster, but seed with the job id so every
  # tasks get the same port
  master_port=$(python3 -c "import random; print(random.Random($SLURM_JOB_ID).randint(10000, 60000))")
  world_size=$(( $SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES ))
  echo "Running on Slurm Cluster, master machine at $master_addr:$master_port"
fi

export EXPERIMENT_DIR=${EXPERIMENT_DIR:-$workspace/experiments}

# for pytorch distributed
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export WORLD_SIZE=$world_size

config=${CONFIG:-""}
resume=${RESUME:-""}

if [[ ($config == "" && $resume == "") || ($config != "" && $resume != "") ]]; then
  echo "Specify either CONFIG or RESUME, but not both or neither"
  exit 1
fi

if [[ $config != "" ]]; then
  echo "Starting training with config $config"
  train_cmd="$code/train.py --config $config"
else
  echo "Resuming training from experiment $resume"
  train_cmd="$code/train.py --resume $resume"
fi

time_out=${TIMEOUT:-23.75h}
if [[ $is_local == true || $force_local == true ]]; then
  echo "Starting local training with cmd $train_cmd"
  timeout -s SIGINT $time_out python3 -W ignore $train_cmd --local
else
  echo "Starting slurm distributed training with cmd $train_cmd"
  srun timeout -s SIGINT $time_out -W ignore $train_cmd
fi
