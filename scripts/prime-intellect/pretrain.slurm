#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --account=research
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=2000:00:00

LM_ENGINE_DIR=""
LM_EVAL_DIR=""
XMA_DIR=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lm-engine-dir)
            LM_ENGINE_DIR="$2"
            shift 2
            ;;
        --lm-eval-dir)
            LM_EVAL_DIR="$2"
            shift 2
            ;;
        --xma-dir)
            XMA_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${LM_ENGINE_DIR}" ]]; then
    echo "[ERROR] --lm-engine-dir is required" >&2
    exit 1
fi

if [[ -z "${LM_EVAL_DIR}" ]]; then
    echo "[ERROR] --lm-eval-dir is required" >&2
    exit 1
fi

if [[ -z "${XMA_DIR}" ]]; then
    echo "[ERROR] --xma-dir is required" >&2
    exit 1
fi

if [[ -z "${CONFIG}" ]]; then
    echo "[ERROR] --config is required" >&2
    exit 1
fi

echo "[INFO] lm-engine-dir: ${LM_ENGINE_DIR}"
echo "[INFO] lm-eval-dir:   ${LM_EVAL_DIR}"
echo "[INFO] xma-dir:       ${XMA_DIR}"
echo "[INFO] config:        ${CONFIG}"

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

REPO_BASE="20260221"
DOCKER_TAG="028a87180e7f9302636da4be3e06f0b4addb76e2"

echo "Running srun torchrun on ${SLURM_JOB_NUM_NODES:-1} node(s)" >&2
echo "Rendezvous endpoint: $MASTER_ADDR:$MASTER_PORT" >&2

srun mkdir -p "/scratch/users/hanguo"

srun \
    --container-image=/data/users/hanguo/images/${REPO_BASE}-${DOCKER_TAG}.sqsh \
    --container-mounts=/data/users/hanguo:/export/share,/data:/export/data,/scratch/users/hanguo:/export/local,${LM_ENGINE_DIR}:/workspace/main/lm-engine,${LM_EVAL_DIR}:/workspace/main/lm-eval-harness,${XMA_DIR}:/workspace/main/accelerated-model-architectures \
    --container-workdir=/workspace/main/lm-engine \
    bash -c "
        pip install -e . && \
        bash /workspace/main/lm-engine/scripts/prime-intellect/install_cutedsl.sh && \
        torchrun \
            --nnodes=$SLURM_JOB_NUM_NODES \
            --nproc_per_node=$SLURM_GPUS_PER_NODE \
            --rdzv_id=$SLURM_JOB_ID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            -m lm_engine.pretrain \
            --config ${CONFIG}
    "
