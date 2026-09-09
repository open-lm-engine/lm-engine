#!/bin/bash
# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

# Adapted from scripts/prime-intellect/train-job.sh for lm_engine.virtual_cluster's `submit`.
#
# Submitted as a real script file (not `sbatch --wrap`) for two reasons:
#   1. `srun torchrun ...` below launches one torchrun process per allocated node, which is
#      what actually forms the multi-node process group. A bare `torchrun` (no srun) inside a
#      --wrap script only ever runs on the job's first node — with --nodes > 1 it just hangs
#      forever waiting for workers that never start.
#   2. A real script file uses its own #!/bin/bash shebang; `--wrap` always uses #!/bin/sh,
#      which on some clusters is dash — no `source` builtin, so venv activation silently fails.
#
# All #SBATCH-able options (--nodes, --gpus-per-node, --chdir, --output, etc.) are passed by
# `submit` on the sbatch command line, not hardcoded here.
#
# Usage: sbatch --nodes=N --gpus-per-node=N --ntasks-per-node=1 --chdir=<workdir> \
#          --output=... --error=... [--partition=...] [--time=...] \
#          train-job.sh <config-path>

export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-$SLURM_GPUS_PER_NODE} / SLURM_GPUS_PER_NODE ))
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SLURM_CPU_BIND=cores
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export TRITON_PRINT_AUTOTUNING=1

# pick up API keys / env vars (e.g. WANDB_API_KEY) exported from your shell rc files — sbatch
# jobs don't get a login/interactive shell, so nothing sources these on its own. Try
# .bash_profile/.profile too, not just .bashrc: a stock .bashrc commonly has an early
# `case $- in *i*) ;; *) return;; esac`-style guard for non-interactive shells, which silently
# no-ops past anything exported below it (move such exports above the guard if this still
# doesn't pick them up).
[ -f ~/.bash_profile ] && source ~/.bash_profile
[ -f ~/.profile ] && source ~/.profile
[ -f ~/.bashrc ] && source ~/.bashrc

[ -f .venv/bin/activate ] && source .venv/bin/activate

# see lm_engine/virtual_cluster/remote.py's slurm_bin(): non-interactive shells on some clusters
# don't have Slurm's bin dir on PATH.
SCONTROL=$(command -v scontrol 2>/dev/null || echo /data/slurm/bin/scontrol)
SRUN=$(command -v srun 2>/dev/null || echo /data/slurm/bin/srun)
MASTER_ADDR=$("$SCONTROL" show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "Running srun torchrun on ${SLURM_JOB_NUM_NODES:-1} node(s)" >&2
echo "Rendezvous endpoint: $MASTER_ADDR:$MASTER_PORT" >&2

"$SRUN" torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m lm_engine.training.train \
    --config ${1}
