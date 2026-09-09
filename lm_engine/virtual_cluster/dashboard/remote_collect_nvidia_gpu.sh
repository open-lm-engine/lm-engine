#!/bin/bash
# Runs on a standalone NVIDIA GPU box (no Slurm — bare host with nvidia-smi).
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
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader 2>&1

echo "@@SECTION:GPUPROCS@@"
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid \
  --format=csv,noheader 2>&1

echo "@@SECTION:PS@@"
ps -eo pid,user,pcpu,pmem,etimes,args --no-headers

echo "@@SECTION:END@@"
