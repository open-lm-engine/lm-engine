#!/bin/bash
# Runs on a Slurm login node. Emits marker-delimited sections for easy parsing.
# Does NOT ssh into compute nodes (not all clusters allow that between nodes).
# Live per-GPU stats are gathered only for jobs the current user owns, via
# `srun --overlap --jobid=X` (attaches to the job's existing allocation,
# doesn't request new resources).
set -o pipefail

find_bin() {
  command -v "$1" 2>/dev/null || echo "/data/slurm/bin/$1"
}
SINFO=$(find_bin sinfo)
SQUEUE=$(find_bin squeue)
SRUN=$(find_bin srun)

echo "@@SECTION:HOST@@"
hostname

echo "@@SECTION:WHOAMI@@"
whoami

echo "@@SECTION:UPTIME@@"
uptime

echo "@@SECTION:MEM@@"
free -m

echo "@@SECTION:DISK@@"
df -h /

echo "@@SECTION:SINFO@@"
"$SINFO" -N -h -o "%N|%P|%t|%G" 2>&1

echo "@@SECTION:SQUEUE@@"
"$SQUEUE" -h -o "%i|%P|%j|%u|%T|%M|%D|%N" 2>&1

echo "@@SECTION:MYGPU@@"
"$SQUEUE" -h -u "$(whoami)" -t RUNNING -o "%i" 2>/dev/null | while read -r jobid; do
  [ -z "$jobid" ] && continue
  echo "### $jobid"
  timeout 8 "$SRUN" --overlap --jobid="$jobid" \
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader 2>&1
done

echo "@@SECTION:END@@"
