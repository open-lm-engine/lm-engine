export TMPDIR=tmp

python tools/wandb_sweep.py \
    --config ${1} \
    --sweep configs/sweep/sweep.yml \
    --slurm_logs_dir sweep \
    --count 100 \
    --max_concurrent 8 \
    --num_nodes 1 \
    --gpus_per_node 8 \
    --project mayank-sweep
