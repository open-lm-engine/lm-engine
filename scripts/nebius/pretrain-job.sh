#!/bin/bash
#SBATCH --job-name=lm-pretrain
#SBATCH --nodes=2                  # <-- change
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=0

export TRITON_PRINT_AUTOTUNING=1
export TOKENIZERS_PARALLELISM=false

# Resolve master node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

torchrun \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_NODEID \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  -m lm_engine.pretrain \
  --config ${1}
