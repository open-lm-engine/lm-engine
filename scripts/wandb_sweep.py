# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""
W&B sweep runner for lm-engine (prime-intellect / Slurm).

Two modes:

1. Create mode (default) — run once on the login node:
   Creates the sweep, then submits --count independent Slurm jobs.
   Each job will pick one trial from the sweep server and run it.

   python scripts/wandb_sweep.py \\
       --config configs/my_config.yaml \\
       --sweep sweep.yaml \\
       --slurm_logs_dir /shared/slurm_logs/my-sweep \\
       --count 10 \\
       [--num_nodes 4] [--gpus_per_node 8] \\
       [--account research] [--time 12:00:00] \\
       [--project my_project] [--entity my_entity]

   To submit more agents to an existing sweep:
       python scripts/wandb_sweep.py --sweep_id <id> --count 5 ...

2. Agent mode (--agent) — run automatically inside each Slurm job:
   Calls wandb.agent() to pick one trial, merges sweep params into the
   base config, writes a temp YAML, and launches srun torchrun.

   python scripts/wandb_sweep.py --agent \\
       --sweep_id <id> --config base.yaml --slurm_logs_dir /shared/...

Sweep YAML (standard W&B format; dot-notation keys map into nested config fields):

    method: bayes
    metric:
      name: train/lm_loss
      goal: minimize
    parameters:
      optimizer_args.class_args.lr:
        distribution: log_uniform_values
        min: 1.0e-5
        max: 1.0e-3
      training_parameters.micro_batch_size:
        values: [4, 8, 16]
"""

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import wandb
import yaml


def _deep_set(d: dict, dotpath: str, value) -> None:
    keys = dotpath.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Agent mode — runs inside each Slurm job
# ---------------------------------------------------------------------------


def _run_as_agent(args) -> None:
    base_config = _load_yaml(args.config)

    # Prefer Slurm env vars (we're inside the job allocation)
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", args.num_nodes))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_PER_NODE", args.gpus_per_node))

    def train_fn():
        # Standard wandb.init() inside the agent — this associates the run
        # with the sweep and loads wandb.config with the trial's parameters.
        run = wandb.init()
        sweep_params = dict(wandb.config)
        run_id = run.id
        project = run.project
        entity = run.entity
        # Finish the driver-side connection; the training subprocess will
        # resume this run via WANDB_RUN_ID + WANDB_RESUME=allow.
        wandb.finish()

        config = copy.deepcopy(base_config)
        for dotpath, value in sweep_params.items():
            _deep_set(config, dotpath, value)

        # Make run name reflect the trial's parameters
        wandb_cfg = config.get("logging_args", {}).get("wandb_args")
        if wandb_cfg is not None:
            base_name = wandb_cfg.get("name", "run")
            suffix = "_".join(f"{k.split('.')[-1]}={v}" for k, v in sweep_params.items())
            wandb_cfg["name"] = f"{base_name}_{suffix}"[:128]

        config_dir = Path(args.slurm_logs_dir) / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        temp_config = str(config_dir / f"sweep-{run_id}.yaml")
        with open(temp_config, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", gpus_per_node * 8))
        env = os.environ.copy()
        env.update(
            {
                "WANDB_RUN_ID": run_id,
                "WANDB_RESUME": "allow",
                "WANDB_PROJECT": project,
                "OMP_NUM_THREADS": str(cpus_per_task // gpus_per_node),
                "MKL_NUM_THREADS": str(cpus_per_task // gpus_per_node),
                "OPENBLAS_NUM_THREADS": str(cpus_per_task // gpus_per_node),
                "NCCL_DEBUG": "WARN",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRITON_PRINT_AUTOTUNING": "1",
                "PYTHONFAULTHANDLER": "1",
            }
        )
        if entity:
            env["WANDB_ENTITY"] = entity

        master_addr = (
            subprocess.check_output(
                ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]],
                text=True,
            )
            .splitlines()[0]
            .strip()
        )

        try:
            cmd = [
                "srun",
                "torchrun",
                f"--nnodes={num_nodes}",
                f"--nproc_per_node={gpus_per_node}",
                "--rdzv_id",
                os.environ["SLURM_JOB_ID"],
                "--rdzv_backend",
                "c10d",
                "--rdzv_endpoint",
                f"{master_addr}:29500",
                "-m",
                "lm_engine.train",
                "--config",
                temp_config,
            ]
            subprocess.run(cmd, env=env, check=True)
        finally:
            os.unlink(temp_config)

    wandb.agent(args.sweep_id, function=train_fn, count=1, project=args.project, entity=args.entity)


# ---------------------------------------------------------------------------
# Create mode — runs on the login node
# ---------------------------------------------------------------------------


def _submit_agent_job(
    sweep_id: str,
    args: argparse.Namespace,
    extra_sbatch_args: list[str],
) -> str:
    """Submit one sbatch job that runs this script in --agent mode."""
    Path(args.slurm_logs_dir).mkdir(parents=True, exist_ok=True)

    agent_parts = [
        sys.executable,
        str(Path(__file__).absolute()),
        "--agent",
        "--sweep_id",
        sweep_id,
        "--config",
        str(Path(args.config).absolute()),
        "--slurm_logs_dir",
        str(Path(args.slurm_logs_dir).absolute()),
        "--gpus_per_node",
        str(args.gpus_per_node),
        "--num_nodes",
        str(args.num_nodes),
    ]
    if args.project:
        agent_parts += ["--project", args.project]
    if args.entity:
        agent_parts += ["--entity", args.entity]

    cmd = [
        "sbatch",
        f"--nodes={args.num_nodes}",
        f"--gpus-per-node={args.gpus_per_node}",
        "--ntasks-per-node=1",
        f"--output={args.slurm_logs_dir}/%x-%j.out",
        f"--error={args.slurm_logs_dir}/%x-%j.err",
        "--wrap",
        " ".join(agent_parts),
    ]
    if args.account:
        cmd.append(f"--account={args.account}")
    if args.time_limit:
        cmd.append(f"--time={args.time_limit}")
    cmd.extend(extra_sbatch_args)

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip().split()[-1]  # "Submitted batch job <id>"


def main():
    parser = argparse.ArgumentParser(
        description="W&B sweep runner for lm-engine (Slurm / prime-intellect)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Base training config YAML path")
    parser.add_argument("--slurm_logs_dir", required=True, help="Shared dir for logs and temp configs")

    # Mode
    parser.add_argument("--agent", action="store_true", help="Run as a wandb agent inside a Slurm job")

    # Sweep
    parser.add_argument("--sweep", default=None, help="Sweep config YAML (required when creating a new sweep)")
    parser.add_argument("--sweep_id", default=None, help="Existing sweep ID (skips sweep creation)")
    parser.add_argument("--count", type=int, default=1, help="Number of Slurm jobs (trials) to submit")

    # Slurm
    parser.add_argument("--num_nodes", type=int, default=1, help="Nodes per job")
    parser.add_argument("--gpus_per_node", type=int, default=8, help="GPUs per node")
    parser.add_argument("--account", default=None, help="Slurm account")
    parser.add_argument("--time", dest="time_limit", default=None, help="Wall-time limit (e.g. 12:00:00)")

    # W&B
    parser.add_argument("--project", default=None, help="W&B project (overrides base config)")
    parser.add_argument("--entity", default=None, help="W&B entity (overrides base config)")

    # Extra args forwarded to sbatch in create mode
    args, extra_sbatch_args = parser.parse_known_args()

    # ---- Agent mode --------------------------------------------------------
    if args.agent:
        if not args.sweep_id:
            parser.error("--sweep_id is required in --agent mode")
        _run_as_agent(args)
        return

    # ---- Create mode -------------------------------------------------------
    if not args.sweep and not args.sweep_id:
        parser.error("--sweep (sweep config YAML) is required when creating a new sweep")

    base_config = _load_yaml(args.config)
    base_wandb = base_config.get("logging_args", {}).get("wandb_args", {}) or {}
    args.project = args.project or base_wandb.get("project")
    args.entity = args.entity or base_wandb.get("entity")

    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_yaml = _load_yaml(args.sweep)
        sweep_id = wandb.sweep(copy.deepcopy(sweep_yaml), project=args.project, entity=args.entity)
        print(f"Created sweep: {sweep_id}")
        if args.entity:
            print(f"  https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")

    args.sweep_id = sweep_id
    job_ids = []
    for _ in range(args.count):
        job_id = _submit_agent_job(sweep_id, args, extra_sbatch_args)
        job_ids.append(job_id)
        print(f"  Submitted job {job_id}")

    print(f"Submitted {len(job_ids)} job(s) for sweep {sweep_id}")


if __name__ == "__main__":
    main()
