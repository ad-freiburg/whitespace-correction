#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=training_tokenization_repair
#SBATCH --mail-user=sebastian.walter98@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00

master_port=${MASTER_PORT:-33334}

force_local=${FORCE_LOCAL:-false}
if [[ -n $SLURM_JOB_ID && $force_local == false ]] ; then
  script_dir=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
  is_local=false
else
  script_dir=$(realpath $0)
  is_local=true
fi
script_dir=$(dirname $script_dir)
workspace=$(realpath $script_dir/../..)
cd $workspace
echo "Script is located at $script_dir, workspace is $workspace"

if [[ $is_local == true ]]; then
  echo "Running locally"
  master_addr="127.0.0.1"
  world_size=$(python -c "import torch; print(torch.cuda.device_count())")
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

  export NCCL_SOCKET_IFNAME=eth0
  export NCCL_IB_DISABLE=1

  master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  world_size=${WORLD_SIZE?"env variable WORLD_SIZE not found"}
  echo "Running on Slurm Cluster, master machine at $master_addr:$master_port"
fi

rsync -ah --progress $LMDB_PATH $TMPDIR/lmdb
export LMDB_PATH=$TMPDIR/lmdb
export EXPERIMENT_DIR=${EXPERIMENT_DIR:-$workspace/experiments}

echo "lmdb path: $LMDB_PATH"

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
  train_cmd="-W ignore -m wsc.train --config $config"
else
  echo "Resuming training from experiment $resume"
  train_cmd="-W ignore -m wsc.train --resume $resume --overwrite-train-data $LMDB_PATH"
fi

if [[ $is_local == true ]]; then
  echo "Starting local training with cmd $train_cmd"
  torchrun \
    --nnodes=1 \
    --nproc_per_node=$world_size \
    $train_cmd
else
  echo "Starting Slurm training with cmd $train_cmd"
  srun python $train_cmd
fi
