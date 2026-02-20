#!/bin/bash
# NOTE: All #SBATCH directives must appear before any non-comment commands.
# -----------------------------------------------------------------------------
# Usage Examples:
#   export VENV_DIR=/shared/venvs/distilbert-train
#   export SLURM_LOGS_DIR=/shared/slurm_logs/distilbert-train
#   sbatch --export=ALL \
#     --output="${SLURM_LOGS_DIR}/%x-%j.out" \
#     --error="${SLURM_LOGS_DIR}/%x-%j.err" \
#     train.slurm
# Or easier way with wrapper script
#   ./submit.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=pt-ddp-train           # Set the job name
#SBATCH --account=research                # Specify the account to charge
#SBATCH --nodes=32                         # Number of nodes to use
#SBATCH --gpus-per-node=8                 # Number of GPUs per node
#SBATCH --ntasks-per-node=1               # Number of tasks per node, meaning one srun command per node
#SBATCH --cpus-per-task=96                # CPU cores per task, per each srun command, 8 cpu cores per GPU
#SBATCH --mem=1000G                        # Memory per node, 10G per CPU core is a good rule of thumb to start with
#SBATCH --time=2000:00:00                   # (2 hours) Wall time limit, days-hours:minutes:seconds
# -----------------------------------------------------------------------------
# NOTE: The --output/--error are set dynamically from submit.sh via command-line:
#   sbatch --export=ALL \
#     --output="${SLURM_LOGS_DIR}/%x-%j.out" \
#     --error="${SLURM_LOGS_DIR}/%x-%j.err" \
#     train.slurm

###########################################################################
# Runtime bootstrap (after directives)                                    #
###########################################################################
# set -euo pipefail

# Validate required env (must be present at submission for SBATCH expansion)
# : "${VENV_DIR:?VENV_DIR not set (export before sbatch)}"
# : "${SLURM_LOGS_DIR:?SLURM_LOGS_DIR not set (export before sbatch)}"

# Source project environment if available
# if [ -f "$VENV_DIR/project.env" ]; then
#     source "$VENV_DIR/project.env"
# fi

# if [ ! -d "$VENV_DIR" ]; then
#     echo "[ERROR] VENV_DIR missing: $VENV_DIR" >&2; exit 12; fi
# if [ ! -f "$VENV_DIR/bin/activate" ]; then
#     echo "[ERROR] venv activate script missing: $VENV_DIR/bin/activate" >&2; exit 13; fi

# source "$VENV_DIR/bin/activate"


export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE ))   # Set  cpus-per-task / gpus-per-node e.g. 64/8=8
export NCCL_DEBUG=WARN        # Enable NCCL debug logging for troubleshooting
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # PyTorch expects this (replaces deprecated NCCL_ASYNC_ERROR_HANDLING warning)
export SLURM_CPU_BIND=cores      # Slurm will bind each task to a set of CPU cores rather than OS (improves performance)
export MKL_NUM_THREADS=$OMP_NUM_THREADS          # Align MKL thread pool with per-process OMP threads to prevent oversubscription
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS     # Keep OpenBLAS thread usage consistent (avoids CPU thrash / context switching)
export PYTHONFAULTHANDLER=1                      # Emit Python traceback immediately on fatal signals/exceptions (faster debugging)
export TOKENIZERS_PARALLELISM=false              # Disable HF tokenizers intra-process thread pool (avoids fork warning & CPU contention)

# Run distributed training using torchrun
# --nnodes: Total number of nodes to use (from Slurm allocation by srun)
# --nproc_per_node: Number of processes (tasks) to launch per node. Standard usage is to set it equal to the number of GPUs per node
# --rdzv_id: Unique rendezvous ID for this job (ensures all nodes join the same group)
# --rdzv_backend: Rendezvous backend for process group initialization (c10d is standard for PyTorch DDP)
# --rdzv_endpoint: Address and port for rendezvous (first Slurm node in allocation, port 29500)
# --max_restarts=2: means it will attempt up to 2 restarts before failing the job if any rank crashes
# Get the first node's hostname for rendezvous

export TRITON_PRINT_AUTOTUNING=1
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500

export WANDB_API_KEY=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
export WANDB_BASE_URL=https://api.wandb.ai

echo "Running srun torchrun on ${SLURM_JOB_NUM_NODES:-1} node(s)" >&2
echo "Rendezvous endpoint: $MASTER_ADDR:$MASTER_PORT" >&2
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m lm_engine.pretrain \
    --config ${1}
