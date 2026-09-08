#!/bin/bash
# Runs on a standalone AMD ROCm GPU box (no Slurm — bare host with rocm-smi).
set -o pipefail

echo "@@SECTION:HOST@@"
hostname

echo "@@SECTION:WHOAMI@@"
whoami

echo "@@SECTION:UPTIME@@"
uptime

echo "@@SECTION:MEM@@"
free -m

echo "@@SECTION:DISK@@"
df -h / 2>&1

echo "@@SECTION:GPUINFO@@"
rocm-smi 2>&1

echo "@@SECTION:GPUPIDS@@"
rocm-smi --showpids 2>&1

echo "@@SECTION:PS@@"
ps -eo pid,user,pcpu,pmem,etimes,args --no-headers

echo "@@SECTION:END@@"
