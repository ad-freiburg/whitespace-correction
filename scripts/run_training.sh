#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --job-name=training
#SBATCH --open-mode=append
#SBATCH --output=train_%x_%j.slurm
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
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

if [[ $is_local == true || $force_local == true ]]; then
  echo "Running locally (force_local=$force_local)"
  master_addr="localhost"
  master_port=$(python3 -c "import random; print(random.Random().randint(10000, 60000))")
  world_size=$(python3 -c "import torch; print(torch.cuda.device_count())")
else
  master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  # set the master port to a random port on the slurm cluster, but seed with the job id so every
  # task gets the same port
  master_port=$(python3 -c "import random; print(random.Random($SLURM_JOB_ID).randint(10000, 60000))")
  world_size=$(( $SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES ))
  echo "Running on Slurm, master machine at $master_addr:$master_port"
fi

# for pytorch distributed
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export WORLD_SIZE=$world_size

config=${CONFIG?"env var CONFIG not found"}
experiment=${EXPERIMENT?"env var EXPERIMENT not found"}

# TODO: change this to the path of your training script
train_script=$(realpath $script_dir/../src/whitespace_correction/api/train.py)
train_cmd="python3 -W ignore $train_script --config $config --experiment $experiment"

# set timeout to something slightly smaller than the jobs time limit
time_out=${TIMEOUT:-23.75h}
if [[ $is_local == true || $force_local == true ]]; then
  echo "Starting local training with cmd $train_cmd"
  train_cmd="$train_cmd --platform local"
  timeout -s SIGINT $time_out $train_cmd
else
  echo "Starting Slurm distributed training with cmd $train_cmd"
  train_cmd="$train_cmd --platform slurm"
  srun timeout -s SIGINT $time_out $train_cmd
fi

# if timeout is reached (exit code 124) and we run on Slurm, 
# requeue the job (training will resume from latest checkpoint)
if [[ $? == 124 && $is_local == false ]]; then
    scontrol requeue $SLURM_JOB_ID
fi
