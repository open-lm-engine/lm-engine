salloc -N 1 \
  --gpus-per-node=8 \
  --ntasks-per-node=1 \
  --cpus-per-task=96 \
  --mem=1000G \
  -t 04:00:00
  srun --pty bash
