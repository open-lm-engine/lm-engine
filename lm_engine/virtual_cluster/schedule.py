# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""
Virtual cluster: a job scheduler + monitoring dashboard for lm-engine
training runs spread across heterogeneous machines (TPU VMs, Slurm/GPU
clusters, bare GPU boxes), all described in one clusters.yaml.

Takes a base training config, layers dataset/load/save-arg overrides on top
of it, ships the merged config to the target cluster over SSH (using the
cluster's `id` as the ssh_host, per ~/.ssh/config), and launches training the
way that cluster's `kind` expects:

    slurm_gpu   -> sbatch (submits and detaches; tracked via a Slurm job id)
    tpu         -> single-process `python -m lm_engine.training.train`,
                   backgrounded with nohup so it survives SSH disconnect
    nvidia_gpu  -> single-node torchrun (nproc-per-node auto-detected),
    amd_gpu        also backgrounded with nohup

An --overrides YAML is keyed by cluster id, one block of overrides per
cluster, so all clusters' dataset/load/save args for an experiment live in a
single file:

    rubin:
      datasets: [...]
      save_args: {save_path: checkpoints/my-run, save_interval: 50}
    sky-b200:
      datasets: [...]
      save_args: {save_path: /data/checkpoints/my-run, save_interval: 50}

Only the block matching --cluster is merged onto --base.

Usage:

    # base config + an overrides YAML keyed by cluster id (see above)
    python -m lm_engine.virtual_cluster submit --cluster rubin \\
        --base configs/pretraining-examples/nvidia-1.yml \\
        --overrides my_overrides.yml \\
        --name my-run

    # or quick dot-path overrides (YAML-parsed values), stackable with --overrides
    python -m lm_engine.virtual_cluster submit --cluster sky-b200 \\
        --base configs/pretraining-examples/nvidia-1.yml \\
        --set save_args.save_path checkpoints/my-run \\
        --set training_parameters.num_training_steps 2000 \\
        --name my-run

    python -m lm_engine.virtual_cluster list-clusters
    python -m lm_engine.virtual_cluster status --cluster rubin --name my-run
    python -m lm_engine.virtual_cluster logs   --cluster rubin --name my-run [--follow]
    python -m lm_engine.virtual_cluster cancel --cluster rubin --name my-run
    python -m lm_engine.virtual_cluster dashboard [--port 8765]

By default clusters are read from lm_engine/virtual_cluster/clusters.yaml
(override with --clusters).
"""

import argparse
import copy
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import yaml


DEFAULT_CLUSTERS_YAML = Path(__file__).parent / "clusters.yaml"
DEFAULT_WORKDIR = "~/lm-engine"
DEFAULT_JOBS_DIR = "~/lm-engine-jobs"
DEFAULT_DASHBOARD_PORT = 8765

_SLURM_KINDS = {"slurm_gpu"}
_TORCHRUN_KINDS = {"nvidia_gpu", "amd_gpu"}
_SINGLE_PROCESS_KINDS = {"tpu"}


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _deep_set(d: dict, dotpath: str, value) -> None:
    keys = dotpath.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _load_clusters(path: str) -> dict[str, dict]:
    data = _load_yaml(path)
    clusters = {}
    for entry in data["clusters"]:
        cid = entry["id"]
        clusters[cid] = {
            "id": cid,
            "label": entry.get("label", cid),
            "kind": entry["kind"],
            "ssh_host": entry.get("ssh_host", cid),
        }
    return clusters


def _get_cluster(args) -> dict:
    clusters = _load_clusters(args.clusters)
    cluster = clusters.get(args.cluster)
    if cluster is None:
        raise SystemExit(f"unknown cluster {args.cluster!r}; known clusters: {', '.join(sorted(clusters))}")
    return cluster


def _ssh(host: str, command: str, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, command]
    return subprocess.run(cmd, capture_output=capture, text=True)


def _scp(local_path: str, host: str, remote_path: str) -> None:
    result = subprocess.run(
        ["scp", "-o", "ConnectTimeout=10", local_path, f"{host}:{remote_path}"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"scp to {host} failed: {result.stderr.strip()}")


def _build_config(args) -> dict:
    config = _load_yaml(args.base)
    for override_path in args.overrides:
        by_cluster = _load_yaml(override_path)
        if args.cluster not in by_cluster:
            raise SystemExit(
                f"{override_path}: no entry for cluster {args.cluster!r} (found: {', '.join(sorted(by_cluster))})"
            )
        config = _deep_merge(config, by_cluster[args.cluster])
    for dotpath, raw_value in args.set:
        _deep_set(config, dotpath, yaml.safe_load(raw_value))
    return config


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def _slurm_launch_command(workdir: str, config_path: str) -> str:
    return (
        f"cd {workdir} && [ -f .venv/bin/activate ] && source .venv/bin/activate; "
        "GPUS_PER_NODE=$(nvidia-smi -L | wc -l); "
        'MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1); '
        "TOKENIZERS_PARALLELISM=false TRITON_PRINT_AUTOTUNING=1 torchrun "
        "--nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$GPUS_PER_NODE "
        "--rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 "
        f"-m lm_engine.training.train --config {config_path}"
    )


def _submit_slurm(host: str, args, job_name: str, remote_job_dir: str, remote_config_path: str) -> str:
    wrap = _slurm_launch_command(args.workdir, remote_config_path)
    sbatch_cmd = (
        f"sbatch --job-name={shlex.quote(job_name)} --nodes={args.nodes} --gpus-per-node={args.gpus_per_node} "
        "--ntasks-per-node=1 "
        + (f"--partition={args.partition} " if args.partition else "")
        + (f"--time={args.time_limit} " if args.time_limit else "")
        + f"--output={remote_job_dir}/train.log --error={remote_job_dir}/train.err "
        + f"--wrap {shlex.quote(wrap)}"
    )

    if args.dry_run:
        print(f"[dry-run] would run on {host}:\n  {sbatch_cmd}")
        return "<dry-run>"

    result = _ssh(host, sbatch_cmd, capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed on {host}: {result.stderr.strip()}")
    job_id = result.stdout.strip().split()[-1]
    _ssh(host, f"echo {job_id} > {remote_job_dir}/jobid", capture=True)
    return job_id


def _background_launch_command(kind: str, workdir: str, config_path: str) -> str:
    activate = "[ -f .venv/bin/activate ] && source .venv/bin/activate; "
    if kind in _TORCHRUN_KINDS:
        run_cmd = (
            "GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l); "
            '[ "$GPUS_PER_NODE" -eq 0 ] 2>/dev/null && GPUS_PER_NODE=$(rocm-smi --showid 2>/dev/null | grep -c "^GPU\\["); '
            "TOKENIZERS_PARALLELISM=false TRITON_PRINT_AUTOTUNING=1 torchrun "
            "--standalone --nnodes=1 --nproc_per_node=$GPUS_PER_NODE "
            f"-m lm_engine.training.train --config {config_path}"
        )
    else:  # tpu, single process
        run_cmd = (
            f"PJRT_DEVICE=TPU TOKENIZERS_PARALLELISM=false python -m lm_engine.training.train --config {config_path}"
        )

    return f"cd {workdir} && {activate}{run_cmd}"


def _submit_background(host: str, args, kind: str, remote_job_dir: str, remote_config_path: str) -> None:
    remote_log_path = f"{remote_job_dir}/train.log"
    inner_cmd = _background_launch_command(kind, args.workdir, remote_config_path)
    remote_cmd = f"nohup bash -c {shlex.quote(inner_cmd)} > {remote_log_path} 2>&1 < /dev/null & disown"

    if args.dry_run:
        print(f"[dry-run] would run on {host}:\n  {remote_cmd}")
        return

    result = _ssh(host, remote_cmd, capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"launch failed on {host}: {result.stderr.strip()}")


def _cmd_submit(args) -> None:
    cluster = _get_cluster(args)
    host = cluster["ssh_host"]
    kind = cluster["kind"]

    config = _build_config(args)
    job_name = args.name or f"{Path(args.base).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    remote_job_dir = f"{args.jobs_dir}/{job_name}"
    remote_config_path = f"{remote_job_dir}/config.yaml"

    print(f"[{cluster['id']}] job {job_name!r} ({kind}) -> {host}:{remote_job_dir}")

    if args.dry_run:
        print("--- merged config ---")
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    else:
        _ssh(host, f"mkdir -p {remote_job_dir}", capture=True)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            local_config_path = f.name
        try:
            _scp(local_config_path, host, remote_config_path)
        finally:
            Path(local_config_path).unlink()

    if kind in _SLURM_KINDS:
        job_id = _submit_slurm(host, args, job_name, remote_job_dir, remote_config_path)
        print(f"  submitted Slurm job {job_id}")
    elif kind in _TORCHRUN_KINDS or kind in _SINGLE_PROCESS_KINDS:
        _submit_background(host, args, kind, remote_job_dir, remote_config_path)
        print("  launched in background (nohup)")
    else:
        raise SystemExit(f"unsupported cluster kind: {kind!r}")

    if not args.dry_run:
        print(f"  config: {host}:{remote_config_path}")
        print(f"  log:    {host}:{remote_job_dir}/train.log")


# ---------------------------------------------------------------------------
# status / logs / cancel
# ---------------------------------------------------------------------------


def _cmd_status(args) -> None:
    cluster = _get_cluster(args)
    host = cluster["ssh_host"]
    remote_job_dir = f"{args.jobs_dir}/{args.name}"

    if cluster["kind"] in _SLURM_KINDS:
        job_id = _ssh(host, f"cat {remote_job_dir}/jobid 2>/dev/null", capture=True).stdout.strip()
        if not job_id:
            print("no Slurm job id recorded for this job")
            return
        result = _ssh(host, f"squeue -j {job_id} -h -o '%i|%T|%M|%N'", capture=True)
        print(result.stdout.strip() or f"job {job_id} not in queue (finished or failed)")
    else:
        remote_config_path = f"{remote_job_dir}/config.yaml"
        result = _ssh(host, f"pgrep -af {shlex.quote(remote_config_path)}", capture=True)
        print(result.stdout.strip() or "not running")


def _cmd_logs(args) -> None:
    cluster = _get_cluster(args)
    host = cluster["ssh_host"]
    remote_log_path = f"{args.jobs_dir}/{args.name}/train.log"
    tail_cmd = f"tail -n {args.lines}" + (" -f" if args.follow else "")
    subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, f"{tail_cmd} {remote_log_path}"])


def _cmd_cancel(args) -> None:
    cluster = _get_cluster(args)
    host = cluster["ssh_host"]
    remote_job_dir = f"{args.jobs_dir}/{args.name}"

    if cluster["kind"] in _SLURM_KINDS:
        job_id = _ssh(host, f"cat {remote_job_dir}/jobid 2>/dev/null", capture=True).stdout.strip()
        if not job_id:
            print("no Slurm job id recorded for this job")
            return
        _ssh(host, f"scancel {job_id}", capture=True)
        print(f"cancelled Slurm job {job_id}")
    else:
        remote_config_path = f"{remote_job_dir}/config.yaml"
        _ssh(host, f"pkill -f {shlex.quote(remote_config_path)}", capture=True)
        print(f"sent SIGTERM to processes running {remote_config_path}")


def _cmd_list_clusters(args) -> None:
    clusters = _load_clusters(args.clusters)
    for cluster in clusters.values():
        print(f"{cluster['id']:<14} {cluster['kind']:<12} ssh_host={cluster['ssh_host']:<14} {cluster['label']}")


def _cmd_dashboard(args) -> None:
    from .dashboard.server import run_dashboard

    run_dashboard(clusters_path=args.clusters, port=args.port)


# ---------------------------------------------------------------------------


def _add_common_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clusters", default=str(DEFAULT_CLUSTERS_YAML), help="path to clusters.yaml")
    parser.add_argument("--cluster", required=True, help="cluster id from clusters.yaml")
    parser.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR, help="remote directory jobs are placed under")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_submit = subparsers.add_parser("submit", help="merge a config and launch a training job on a cluster")
    _add_common_cluster_args(p_submit)
    p_submit.add_argument("--base", required=True, help="base training config YAML")
    p_submit.add_argument(
        "--overrides",
        action="append",
        default=[],
        help="overrides YAML keyed by cluster id (only the --cluster's entry is merged onto --base); repeatable",
    )
    p_submit.add_argument(
        "--set",
        nargs=2,
        metavar=("DOTPATH", "VALUE"),
        action="append",
        default=[],
        help="dot-path override, e.g. --set save_args.save_path checkpoints/run1",
    )
    p_submit.add_argument("--name", default=None, help="job name (default: <base config stem>-<timestamp>)")
    p_submit.add_argument("--workdir", default=DEFAULT_WORKDIR, help="remote lm-engine checkout to run from")
    p_submit.add_argument("--nodes", type=int, default=1, help="[slurm_gpu] nodes to request")
    p_submit.add_argument("--gpus-per-node", type=int, default=8, help="[slurm_gpu] GPUs per node to request")
    p_submit.add_argument("--partition", default=None, help="[slurm_gpu] Slurm partition")
    p_submit.add_argument("--time", dest="time_limit", default=None, help="[slurm_gpu] wall-time limit, e.g. 12:00:00")
    p_submit.add_argument(
        "--dry-run", action="store_true", help="print the merged config and launch command without doing anything"
    )
    p_submit.set_defaults(func=_cmd_submit)

    p_status = subparsers.add_parser("status", help="check whether a job is running")
    _add_common_cluster_args(p_status)
    p_status.add_argument("--name", required=True, help="job name")
    p_status.set_defaults(func=_cmd_status)

    p_logs = subparsers.add_parser("logs", help="tail a job's training log")
    _add_common_cluster_args(p_logs)
    p_logs.add_argument("--name", required=True, help="job name")
    p_logs.add_argument("--lines", type=int, default=200, help="number of trailing lines to show")
    p_logs.add_argument("--follow", action="store_true", help="keep streaming new log lines (like tail -f)")
    p_logs.set_defaults(func=_cmd_logs)

    p_cancel = subparsers.add_parser("cancel", help="stop a running job")
    _add_common_cluster_args(p_cancel)
    p_cancel.add_argument("--name", required=True, help="job name")
    p_cancel.set_defaults(func=_cmd_cancel)

    p_list = subparsers.add_parser("list-clusters", help="list clusters from clusters.yaml")
    p_list.add_argument("--clusters", default=str(DEFAULT_CLUSTERS_YAML), help="path to clusters.yaml")
    p_list.set_defaults(func=_cmd_list_clusters)

    p_dashboard = subparsers.add_parser("dashboard", help="serve the virtual cluster monitoring dashboard")
    p_dashboard.add_argument("--clusters", default=str(DEFAULT_CLUSTERS_YAML), help="path to clusters.yaml")
    p_dashboard.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT, help="port to serve on")
    p_dashboard.set_defaults(func=_cmd_dashboard)

    args = parser.parse_args()
    args.func(args)
