#!/bin/bash

set -euo pipefail

NAME=""
CONFIG=""
MAX_LENGTH=""
RESUME=false


while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${NAME}" ]]; then
    echo "[ERROR] --name is required" >&2
    exit 1
fi

if [[ -z "${CONFIG}" ]]; then
    echo "[ERROR] --config is required" >&2
    exit 1
fi

if [[ -z "${MAX_LENGTH}" ]]; then
    echo "[ERROR] --max-length is required" >&2
    exit 1
fi

echo "[INFO] name:       ${NAME}"
echo "[INFO] config:     ${CONFIG}"
echo "[INFO] max-length: ${MAX_LENGTH}"
echo "[INFO] resume:     ${RESUME}"

EXPERIMENT_DIR="/data/users/hanguo/experiments/${NAME}"
LM_ENGINE_DIR="${EXPERIMENT_DIR}/lm-engine"
LM_EVAL_DIR="${EXPERIMENT_DIR}/lm-eval-harness"
XMA_DIR="${EXPERIMENT_DIR}/accelerated-model-architectures"

if [[ "${RESUME}" == true ]]; then
    if [[ ! -d "${EXPERIMENT_DIR}" ]]; then
        echo "[ERROR] Experiment directory not found for resume: ${EXPERIMENT_DIR}" >&2
        exit 1
    fi
    echo "[INFO] Resuming experiment in ${EXPERIMENT_DIR}"
else
    if [[ -d "${EXPERIMENT_DIR}" ]]; then
        echo "[ERROR] Experiment directory already exists: ${EXPERIMENT_DIR}" >&2
        exit 1
    fi

    mkdir -p "${EXPERIMENT_DIR}"
    echo "[INFO] Copying lm-engine → ${LM_ENGINE_DIR}"
    cp -r "$(pwd)" "${LM_ENGINE_DIR}"
    echo "[INFO] Copying lm-eval-harness → ${LM_EVAL_DIR}"
    cp -r "$(pwd)/../lm-eval-harness" "${LM_EVAL_DIR}"
    echo "[INFO] Copying accelerated-model-architectures → ${XMA_DIR}"
    cp -r "$(pwd)/../accelerated-model-architectures" "${XMA_DIR}"
fi

TRAIN_JOB_ID=$(
    sbatch \
        --parsable \
        "${LM_ENGINE_DIR}/scripts/prime-intellect/pretrain.slurm" \
            --lm-engine-dir "${LM_ENGINE_DIR}" \
            --lm-eval-dir "${LM_EVAL_DIR}" \
            --xma-dir "${XMA_DIR}" \
            --config "${CONFIG}"
)

echo "[INFO] Training job submitted: ${TRAIN_JOB_ID}"

EVAL_JOB_ID=$(
    sbatch \
        --parsable \
        --kill-on-invalid-dep=yes \
        --dependency=afterok:"${TRAIN_JOB_ID}" \
        "${LM_ENGINE_DIR}/scripts/prime-intellect/eval.slurm" \
            --lm-engine-dir "${LM_ENGINE_DIR}" \
            --lm-eval-dir "${LM_EVAL_DIR}" \
            --xma-dir "${XMA_DIR}" \
            --max-length "${MAX_LENGTH}"
)

echo "[INFO] Eval job submitted: ${EVAL_JOB_ID} (depends on ${TRAIN_JOB_ID})"
