# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""
W&B sweep runner for lm-engine (prime-intellect / Slurm).

Creates a W&B sweep from a base training config and a sweep config, then runs
an agent that submits one Slurm batch job per trial via train-job.sh and waits
for it to finish before requesting the next set of parameters.

Usage:
    python scripts/wandb_sweep.py \\
        --config configs/my_config.yaml \\
        --sweep sweep.yaml \\
        --slurm_logs_dir /shared/slurm_logs/my-sweep \\
        [--num_nodes 4] \\
        [--gpus_per_node 8] \\
        [--account research] \\
        [--time 12:00:00] \\
        [--count 20] \\
        [--project my_project] \\
        [--entity my_entity] \\
        [--sweep_id existing-sweep-id] \\
        [--poll_interval 60]

Sweep YAML format (W&B sweep config with dot-notation parameter keys that map
to nested fields in the base training config):

    method: bayes
    metric:
      name: val/loss
      goal: minimize
    parameters:
      optimizer_args.class_args.lr:
        distribution: log_uniform_values
        min: 1.0e-5
        max: 1.0e-3
      training_parameters.micro_batch_size:
        values: [4, 8, 16]
      optimizer_args.class_args.weight_decay:
        values: [0.01, 0.1, 0.3]

Parameter keys use dot-notation to reference nested config fields.
W&B fields (project, entity) in the sweep YAML override the base config.
The temp per-trial config is written to --slurm_logs_dir so it is accessible
from compute nodes (must be on a shared filesystem).
"""

import argparse
import copy
import os
import subprocess
import time
from pathlib import Path

import wandb
import yaml


_SCRIPT_DIR = Path(__file__).parent
_TRAIN_JOB_SCRIPT = _SCRIPT_DIR / "prime-intellect" / "train-job.sh"


def _deep_set(d: dict, dotpath: str, value) -> None:
    keys = dotpath.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _submit_job(
    config_path: str,
    num_nodes: int,
    gpus_per_node: int,
    slurm_logs_dir: str,
    account: str | None,
    time_limit: str | None,
    wandb_env: dict[str, str],
    extra_sbatch_args: list[str],
) -> str:
    """Submit train-job.sh via sbatch and return the Slurm job ID."""
    Path(slurm_logs_dir).mkdir(parents=True, exist_ok=True)

    export_pairs = ",".join(f"{k}={v}" for k, v in wandb_env.items())

    cmd = [
        "sbatch",
        f"--nodes={num_nodes}",
        f"--gpus-per-node={gpus_per_node}",
        f"--output={slurm_logs_dir}/%x-%j.out",
        f"--error={slurm_logs_dir}/%x-%j.err",
        f"--export=ALL,{export_pairs}",
    ]
    if account:
        cmd.append(f"--account={account}")
    if time_limit:
        cmd.append(f"--time={time_limit}")
    cmd.extend(extra_sbatch_args)
    cmd.extend([str(_TRAIN_JOB_SCRIPT), config_path])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # sbatch prints "Submitted batch job <job_id>"
    job_id = result.stdout.strip().split()[-1]
    return job_id


def _wait_for_job(job_id: str, poll_interval: int) -> None:
    """Block until the Slurm job is no longer in the queue."""
    while True:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
        )
        state = result.stdout.strip()
        if not state:
            break
        print(f"  job {job_id}: {state}", flush=True)
        time.sleep(poll_interval)


def _make_train_fn(
    base_config: dict,
    num_nodes: int,
    gpus_per_node: int,
    slurm_logs_dir: str,
    account: str | None,
    time_limit: str | None,
    poll_interval: int,
    extra_sbatch_args: list[str],
):
    def train_fn():
        run = wandb.init()
        sweep_params = dict(wandb.config)
        run_id = run.id
        project = run.project
        entity = run.entity
        wandb.finish()

        config = copy.deepcopy(base_config)
        for dotpath, value in sweep_params.items():
            _deep_set(config, dotpath, value)

        # Append sweep param summary to the run name so trials are distinguishable
        wandb_args = config.get("logging_args", {}).get("wandb_args")
        if wandb_args is not None:
            base_name = wandb_args.get("name", "run")
            suffix = "_".join(f"{k.split('.')[-1]}={v}" for k, v in sweep_params.items())
            wandb_args["name"] = f"{base_name}_{suffix}"[:128]

        # Write temp config to the shared logs dir so compute nodes can read it
        config_dir = Path(slurm_logs_dir) / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        temp_config = str(config_dir / f"sweep-{run_id}.yaml")
        with open(temp_config, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        wandb_env = {
            "WANDB_RUN_ID": run_id,
            "WANDB_RESUME": "allow",
            "WANDB_PROJECT": project,
        }
        if entity:
            wandb_env["WANDB_ENTITY"] = entity

        try:
            job_id = _submit_job(
                temp_config,
                num_nodes,
                gpus_per_node,
                slurm_logs_dir,
                account,
                time_limit,
                wandb_env,
                extra_sbatch_args,
            )
            print(f"Submitted job {job_id} (run {run_id})", flush=True)
            _wait_for_job(job_id, poll_interval)
        finally:
            os.unlink(temp_config)

    return train_fn


def main():
    parser = argparse.ArgumentParser(
        description="Create and run a W&B sweep for lm-engine via Slurm (prime-intellect)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Base training config YAML path")
    parser.add_argument("--sweep", required=True, help="W&B sweep config YAML path")
    parser.add_argument(
        "--slurm_logs_dir", required=True, help="Shared directory for Slurm stdout/stderr logs and temp configs"
    )
    parser.add_argument("--sweep_id", default=None, help="Existing sweep ID to resume (skips sweep creation)")
    parser.add_argument("--count", type=int, default=None, help="Max runs for this agent (default: unlimited)")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes (--nodes sbatch override)")
    parser.add_argument("--gpus_per_node", type=int, default=8, help="GPUs per node (--gpus-per-node sbatch override)")
    parser.add_argument("--account", default=None, help="Slurm account (--account)")
    parser.add_argument(
        "--time", dest="time_limit", default=None, help="Wall-time limit passed to sbatch (e.g. 12:00:00)"
    )
    parser.add_argument("--poll_interval", type=int, default=60, help="Seconds between squeue polls (default: 60)")
    parser.add_argument("--project", default=None, help="W&B project (overrides base config and sweep YAML)")
    parser.add_argument("--entity", default=None, help="W&B entity (overrides base config and sweep YAML)")
    # Unknown args are forwarded verbatim to sbatch
    args, extra_sbatch_args = parser.parse_known_args()

    base_config = _load_yaml(args.config)
    sweep_yaml = _load_yaml(args.sweep)

    # Resolve project/entity: CLI > sweep YAML > base config
    base_wandb = base_config.get("logging_args", {}).get("wandb_args", {}) or {}
    project = args.project or sweep_yaml.get("project") or base_wandb.get("project")
    entity = args.entity or sweep_yaml.get("entity") or base_wandb.get("entity")

    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_config = copy.deepcopy(sweep_yaml)
        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        print(f"Created sweep: {sweep_id}")
        if entity:
            print(f"View at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")

    train_fn = _make_train_fn(
        base_config,
        args.num_nodes,
        args.gpus_per_node,
        args.slurm_logs_dir,
        args.account,
        args.time_limit,
        args.poll_interval,
        extra_sbatch_args,
    )

    wandb.agent(sweep_id, function=train_fn, count=args.count, project=project, entity=entity)


if __name__ == "__main__":
    main()
