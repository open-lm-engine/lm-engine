salloc -N 1 \
  --gpus-per-node=8 \
  --cpus-per-task=128 \
  --mem=0 \
  -t 04:00:00

srun --nodes=1 --ntasks=1 --pty bash
