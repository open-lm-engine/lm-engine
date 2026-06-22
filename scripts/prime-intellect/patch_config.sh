#!/bin/bash

CONFIG="configs/research/delta-mlp/delta-mlp-64k.yml"
NAME=""
KEY=""
VALUE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --key)
            KEY="$2"
            shift 2
            ;;
        --value)
            VALUE="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${CONFIG}" ]]; then
    echo "[ERROR] --config is required" >&2
    exit 1
fi

if [[ -z "${NAME}" ]]; then
    echo "[ERROR] --name is required" >&2
    exit 1
fi

if [[ -z "${KEY}" ]]; then
    echo "[ERROR] --key is required" >&2
    exit 1
fi

if [[ -z "${VALUE}" ]]; then
    echo "[ERROR] --value is required" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG}" >&2
    exit 1
fi

# Patch key-value pair (replaces all occurrences at any indentation)
sed -i "s/^\([[:space:]]*${KEY}:[[:space:]]*\).*/\1${VALUE}/" "${CONFIG}"
echo "[INFO] Patched ${KEY}: ${VALUE} in ${CONFIG}" >&2

# Patch wandb run name
sed -i "s/^\([[:space:]]*name:[[:space:]]*\).*/\1${NAME}/" "${CONFIG}"
echo "[INFO] Patched wandb name: ${NAME} in ${CONFIG}" >&2
