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
   Registers with the sweep server via direct HTTP (no wandb service socket),
   fetches the trial's hyperparameters, merges them into the base config,
   writes a temp YAML, and launches srun torchrun.

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
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
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
# W&B sweep API — direct HTTP GraphQL (no wandb service socket required)
#
# wandb.agent() routes all API calls through a local service daemon socket.
# On HPC clusters the socket is created on the login node; compute nodes
# inherit the socket path via SLURM env vars but can't reach it, causing
# WandbServiceConnectionError.  These helpers talk to the W&B API directly
# over HTTPS, bypassing the service entirely.
# ---------------------------------------------------------------------------

_REGISTER_AGENT_MUTATION = """
mutation CreateAgent($input: CreateAgentInput!) {
    createAgent(input: $input) {
        agent { id }
    }
}
"""

_AGENT_HEARTBEAT_MUTATION = """
mutation AgentHeartbeat($id: ID!, $metrics: JSONString, $runState: JSONString) {
    agentHeartbeat(input: {id: $id, metrics: $metrics, runState: $runState}) {
        commands
    }
}
"""


def _wandb_graphql(api_key: str, query: str, variables: dict) -> dict:
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    resp = requests.post(
        f"{base_url}/graphql",
        auth=(api_key, api_key),
        json={"query": query, "variables": variables},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if errors := payload.get("errors"):
        raise RuntimeError(f"W&B API error: {errors}")
    return payload["data"]


def _sweep_register_agent(api_key: str, entity: str | None, project: str | None, sweep_id: str) -> str:
    import socket as _socket

    data = _wandb_graphql(
        api_key,
        _REGISTER_AGENT_MUTATION,
        {
            "input": {
                "host": _socket.gethostname(),
                "sweep": sweep_id,
                "projectName": project,
                "entityName": entity,
            }
        },
    )
    return data["createAgent"]["agent"]["id"]


def _sweep_next_run(api_key: str, agent_id: str, max_polls: int = 30) -> tuple[str | None, dict]:
    """Poll sweep server; returns (run_id, {param: value}) or (None, {}) when done."""
    for _ in range(max_polls):
        data = _wandb_graphql(
            api_key,
            _AGENT_HEARTBEAT_MUTATION,
            {"id": agent_id, "metrics": json.dumps({}), "runState": json.dumps({})},
        )
        commands_str = data["agentHeartbeat"]["commands"]
        commands: list[dict] = json.loads(commands_str) if commands_str else []
        for cmd in commands:
            if cmd.get("type") == "run":
                run_id = cmd.get("run_id") or cmd.get("runId")
                raw_args: dict = cmd.get("args", {})
                # W&B encodes each param as {"value": v}; unwrap if present.
                params = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in raw_args.items()}
                return run_id, params
            if cmd.get("type") in ("stop", "exit"):
                return None, {}
        time.sleep(2)
    return None, {}


# ---------------------------------------------------------------------------
# Agent mode — runs inside each Slurm job
# ---------------------------------------------------------------------------


def _run_as_agent(args) -> None:
    base_config = _load_yaml(args.config)

    # Prefer Slurm env vars (we're inside the job allocation)
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", args.num_nodes))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_PER_NODE", args.gpus_per_node))
    tmpdir = os.environ.get("TMPDIR", "")

    base_wandb_cfg = base_config.get("logging_args", {}).get("wandb_args") or {}
    entity = args.entity or base_wandb_cfg.get("entity")
    project = args.project or base_wandb_cfg.get("project")

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY environment variable is not set")

    # Register agent and fetch next run config via direct HTTP (no service socket).
    agent_id = _sweep_register_agent(api_key, entity, project, args.sweep_id)
    run_id, sweep_params = _sweep_next_run(api_key, agent_id)

    if run_id is None:
        print("Sweep is done or no run was assigned; exiting.")
        return

    config = copy.deepcopy(base_config)
    for dotpath, value in sweep_params.items():
        _deep_set(config, dotpath, value)

    # Let wandb assign its default auto-generated name for each sweep trial.
    wandb_cfg = config.get("logging_args", {}).get("wandb_args")
    if wandb_cfg is not None:
        wandb_cfg.pop("name", None)

    config_dir = Path(args.slurm_logs_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    temp_config = str(config_dir / f"sweep-{run_id}.yaml")
    with open(temp_config, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # SLURM_CPUS_PER_TASK and SLURM_TRES_PER_TASK can conflict when both are
    # set (srun treats it as a fatal error).  Derive cpus_per_task from TRES
    # first; fall back to SLURM_CPUS_PER_TASK only when TRES is absent.
    def _parse_tres_cpus(tres: str) -> int | None:
        for part in tres.split(","):
            if part.startswith("cpu="):
                return int(part.split("=", 1)[1])
        return None

    tres = os.environ.get("SLURM_TRES_PER_TASK", "")
    cpus_per_task = _parse_tres_cpus(tres) or int(os.environ.get("SLURM_CPUS_PER_TASK", gpus_per_node * 8))

    env = os.environ.copy()
    env.pop("SLURM_CPUS_PER_TASK", None)  # let srun derive from SLURM_TRES_PER_TASK
    # Scrub any wandb service socket vars inherited from the login node via SLURM.
    # If present they point to a socket that doesn't exist on compute nodes, causing
    # WandbServiceConnectionError even when WANDB__DISABLE_SERVICE=true is set.
    for _var in [k for k in env if k.startswith("WANDB_SERVICE") or k == "_WANDB_STARTUP_DEBUG"]:
        env.pop(_var)
    env.update(
        {
            "WANDB_RUN_ID": run_id,
            "WANDB_RESUME": "allow",
            "WANDB_SWEEP_ID": args.sweep_id,
            # "WANDB__DISABLE_SERVICE": "true",
            "OMP_NUM_THREADS": str(cpus_per_task // gpus_per_node),
            "MKL_NUM_THREADS": str(cpus_per_task // gpus_per_node),
            "OPENBLAS_NUM_THREADS": str(cpus_per_task // gpus_per_node),
            "NCCL_DEBUG": "WARN",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRITON_PRINT_AUTOTUNING": "1",
            "PYTHONFAULTHANDLER": "1",
            "TMPDIR": tmpdir,
        }
    )
    if entity:
        env["WANDB_ENTITY"] = entity
    if project:
        env["WANDB_PROJECT"] = project

    master_addr = (
        subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]],
            text=True,
        )
        .splitlines()[0]
        .strip()
    )

    torchrun_args = [
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
    # srun is only needed for multi-node to launch one torchrun per node.
    # For single-node it fights with the batch step for the task slot.
    if num_nodes > 1:
        cmd = ["srun", "--overlap"] + torchrun_args
    else:
        cmd = torchrun_args

    try:
        subprocess.run(cmd, env=env, check=True)
    finally:
        os.unlink(temp_config)


# ---------------------------------------------------------------------------
# Create mode — runs on the login node
# ---------------------------------------------------------------------------


def _count_running_jobs(job_ids: list[str]) -> int:
    """Return how many of the given Slurm job IDs are still in the queue."""
    if not job_ids:
        return 0
    result = subprocess.run(
        ["squeue", "--jobs", ",".join(job_ids), "-h", "-o", "%i"],
        capture_output=True,
        text=True,
    )
    return len(result.stdout.strip().splitlines())


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
        f"--mem-per-gpu={args.mem_per_gpu}G",
        f"--cpus-per-gpu={args.cpus_per_gpu}",
        "--ntasks-per-node=1",
        f"--output={args.slurm_logs_dir}/%x-%j.out",
        f"--error={args.slurm_logs_dir}/%x-%j.err",
        "--wrap",
        " ".join(agent_parts),
    ]
    if args.num_nodes == 1:
        # Allow the scheduler to pack this job onto an already-occupied node
        # rather than allocating a new one exclusively.
        cmd.append("--oversubscribe")
    if args.account:
        cmd.append(f"--account={args.account}")
    if args.time_limit:
        cmd.append(f"--time={args.time_limit}")
    cmd.extend(extra_sbatch_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed (exit {result.returncode}):\n{result.stderr.strip()}")
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
    parser.add_argument("--count", type=int, default=1, help="Total number of trials to run")
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=None,
        help="Max jobs running at once; new jobs are submitted as slots free up (default: all at once)",
    )

    # Slurm
    parser.add_argument("--num_nodes", type=int, default=1, help="Nodes per job")
    parser.add_argument("--gpus_per_node", type=int, default=8, help="GPUs per node")
    parser.add_argument("--mem_per_gpu", type=int, default=120, help="System RAM per GPU in GB (e.g. 4 GPUs → 480 GB)")
    parser.add_argument("--cpus_per_gpu", type=int, default=12, help="CPU cores per GPU")
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

    _DEFAULT_METRIC = {"name": "train/lm_loss", "goal": "minimize"}

    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_yaml = _load_yaml(args.sweep)
        sweep_config = copy.deepcopy(sweep_yaml)
        sweep_config.setdefault("metric", _DEFAULT_METRIC)
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        print(f"Created sweep: {sweep_id}")
        if args.entity:
            print(f"  https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")

    args.sweep_id = sweep_id
    max_concurrent = args.max_concurrent or args.count
    active_job_ids: list[str] = []
    total_submitted = 0

    while total_submitted < args.count:
        running = _count_running_jobs(active_job_ids)
        slots = max_concurrent - running
        for _ in range(min(slots, args.count - total_submitted)):
            job_id = _submit_agent_job(sweep_id, args, extra_sbatch_args)
            active_job_ids.append(job_id)
            total_submitted += 1
            print(f"  Submitted job {job_id} ({total_submitted}/{args.count})")
        if total_submitted < args.count:
            time.sleep(5)

    print(f"Submitted {total_submitted} job(s) for sweep {sweep_id}")


if __name__ == "__main__":
    main()
