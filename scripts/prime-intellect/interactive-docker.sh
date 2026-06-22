salloc -N 1 \
    --gpus-per-node=8 \
    --cpus-per-task=128 \
    --mem=1000G \
    -t 10-0

REPO_BASE="20260221"
DOCKER_TAG="028a87180e7f9302636da4be3e06f0b4addb76e2"
EXPERIMENT_DIR="/data/users/hanguo/experiments/interactive-0"
LM_ENGINE_DIR="${EXPERIMENT_DIR}/lm-engine"
LM_EVAL_DIR="${EXPERIMENT_DIR}/lm-eval-harness"
XMA_DIR="${EXPERIMENT_DIR}/accelerated-model-architectures"

srun \
    --container-image=/data/users/hanguo/images/${REPO_BASE}-${DOCKER_TAG}.sqsh \
    --container-mounts=/data/users/hanguo:/export/share,/data:/export/data,/scratch/users/hanguo:/export/local,${LM_ENGINE_DIR}:/workspace/main/lm-engine,${LM_EVAL_DIR}:/workspace/main/lm-eval-harness,${XMA_DIR}:/workspace/main/accelerated-model-architectures \
    --container-workdir=/workspace/main/lm-engine \
    --nodes=1 \
    --ntasks=1 \
    bash -c "
        pip install -e . && \
        pip install -e '/workspace/main/lm-eval-harness[hf]' && \
        pip install -e /workspace/main/accelerated-model-architectures && \
        bash /workspace/main/lm-engine/scripts/prime-intellect/install_cutedsl.sh && \
        jupyter-lab --ip=0.0.0.0
    "
