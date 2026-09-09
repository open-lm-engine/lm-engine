#!/bin/bash
# Runs on the TPU host. Emits marker-delimited sections for easy parsing.
set -o pipefail

echo "@@SECTION:HOST@@"
hostname

echo "@@SECTION:ACCEL@@"
curl -s -m 3 -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type" 2>/dev/null

echo
echo "@@SECTION:UPTIME@@"
uptime

echo "@@SECTION:MEM@@"
free -m

echo "@@SECTION:DISK@@"
df -h /

echo "@@SECTION:TPUINFO@@"
VENV=/home/mayank/scratch/lm-engine/.venv
if [ -x "$VENV/bin/tpu-info" ]; then
  "$VENV/bin/tpu-info" 2>&1
else
  echo "TPU_INFO_UNAVAILABLE"
fi

echo "@@SECTION:PS@@"
ps -eo pid,user,pcpu,pmem,etimes,args --no-headers

echo "@@SECTION:END@@"
