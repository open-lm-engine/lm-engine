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

Pass --cluster auto instead of a specific id to have it check every
cluster's live status (same probes the dashboard uses — idle Slurm nodes,
or no process running on a bare box/TPU VM) and submit to whichever one has
free capacity right now. This is a one-shot "wherever's free right now"
pick, not a queue that waits for a slot to open up.

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

    # let the scheduler pick whichever cluster is free right now
    python -m lm_engine.virtual_cluster submit --cluster auto \\
        --base configs/pretraining-examples/nvidia-1.yml \\
        --overrides my_overrides.yml \\
        --name my-run

    python -m lm_engine.virtual_cluster list-clusters
    python -m lm_engine.virtual_cluster status --cluster rubin --name my-run
    python -m lm_engine.virtual_cluster logs   --cluster rubin --name my-run [--follow]
    python -m lm_engine.virtual_cluster cancel --cluster rubin --name my-run
    python -m lm_engine.virtual_cluster dashboard [--port 8765]

`status`/`logs`/`cancel` above also work for W&B sweeps launched with
`tools/wandb_sweep.py --cluster ...` (see that script's --help) — both tools
lay out remote job directories the same way (a `kind` marker plus, for
`submit`, a `workdir` marker), so this is a single place to check on
anything running on the virtual cluster, regardless of how it was launched.
A `submit`-launched training job's actual output lives at
<workdir>/logs/<id>-out.log and -err.log (<id> is the Slurm job id, or the
job name on a bare box/TPU VM); a sweep's driver log is still under its own
--jobs-dir job directory as run.log.

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

from .remote import (
    DEFAULT_CLUSTERS_YAML,
    DEFAULT_JOBS_DIR,
    DEFAULT_WORKDIR,
    check_node_limit,
    ensure_job_dir,
    get_cluster,
    load_clusters,
    new_job_dir,
    resolve_workdir,
    scp,
    slurm_bin,
    ssh,
)


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


TRAIN_JOB_SCRIPT = Path(__file__).parent / "train-job.sh"


def _submit_slurm(host: str, args, job_name: str, remote_job_dir: str, remote_config_path: str) -> str:
    remote_script_path = f"{remote_job_dir}/train-job.sh"
    logs_dir = f"{args.workdir}/logs"
    # Slurm substitutes %j with the actual job id once assigned — we don't know it before
    # submitting, so this is the only way to name the files after it.
    out_path = f"{logs_dir}/%j-out.log"
    err_path = f"{logs_dir}/%j-err.log"

    if not args.dry_run:
        ssh(host, f"mkdir -p {logs_dir}", capture=True)
        scp(str(TRAIN_JOB_SCRIPT), host, remote_script_path)

    sbatch_cmd = (
        f"{slurm_bin('sbatch')} --job-name={shlex.quote(job_name)} --nodes={args.nodes} --gpus-per-node={args.gpus_per_node} "
        # --chdir, the log paths, and the trailing script/config args are left unquoted (like
        # remote_job_dir elsewhere) so a `~`-based workdir still gets expanded by the remote shell.
        f"--ntasks-per-node=1 --chdir={args.workdir} "
        + (f"--partition={args.partition} " if args.partition else "")
        + (f"--time={args.time_limit} " if args.time_limit else "")
        + f"--output={out_path} --error={err_path} "
        + f"{remote_script_path} {remote_config_path}"
    )

    if args.dry_run:
        print(f"[dry-run] would upload {TRAIN_JOB_SCRIPT} -> {host}:{remote_script_path}")
        print(f"[dry-run] would run on {host}:\n  {sbatch_cmd}")
        return "<dry-run>"

    result = ssh(host, sbatch_cmd, capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed on {host}: {result.stderr.strip()}")
    job_id = result.stdout.strip().split()[-1]
    ssh(host, f"echo {job_id} > {remote_job_dir}/jobid", capture=True)
    return job_id


def _background_launch_command(kind: str, workdir: str, config_path: str) -> str:
    # non-interactive nohup'd shell won't source any rc file on its own, so API keys/env vars
    # kept there (e.g. WANDB_API_KEY) wouldn't otherwise be visible to the training process. Try
    # .bash_profile/.profile too: a stock .bashrc commonly has an early
    # `case $- in *i*) ;; *) return;; esac`-style guard for non-interactive shells that silently
    # no-ops past anything exported below it.
    bashrc = "[ -f ~/.bash_profile ] && source ~/.bash_profile; [ -f ~/.profile ] && source ~/.profile; [ -f ~/.bashrc ] && source ~/.bashrc; "
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

    return f"cd {workdir} && {bashrc}{activate}{run_cmd}"


def _submit_background(host: str, args, kind: str, job_name: str, remote_config_path: str) -> None:
    logs_dir = f"{args.workdir}/logs"
    out_path = f"{logs_dir}/{job_name}-out.log"
    err_path = f"{logs_dir}/{job_name}-err.log"
    inner_cmd = _background_launch_command(kind, args.workdir, remote_config_path)
    remote_cmd = f"mkdir -p {logs_dir} && nohup bash -c {shlex.quote(inner_cmd)} > {out_path} 2> {err_path} < /dev/null & disown"

    if args.dry_run:
        print(f"[dry-run] would run on {host}:\n  {remote_cmd}")
        return

    result = ssh(host, remote_cmd, capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"launch failed on {host}: {result.stderr.strip()}")


def _cmd_submit(args) -> None:
    cluster = get_cluster(args.clusters, args.cluster)
    host = cluster["ssh_host"]
    kind = cluster["kind"]
    check_node_limit(cluster, args.nodes)
    args.workdir = resolve_workdir(cluster, args.workdir)

    config = _build_config(args)
    job_name = args.name or f"{Path(args.base).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    # hash-suffixed so a resubmit under the same --name never clobbers a still-in-flight
    # submission's config/logs; <jobs_dir>/<job_name> is kept pointed at the latest one below.
    remote_job_dir = new_job_dir(args.jobs_dir, job_name)
    remote_config_path = f"{remote_job_dir}/config.yaml"

    print(f"[{cluster['id']}] job {job_name!r} ({kind}) -> {host}:{remote_job_dir}")

    if args.dry_run:
        print("--- merged config ---")
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    else:
        ensure_job_dir(host, args.jobs_dir, job_name, remote_job_dir)
        ssh(host, f"echo train > {remote_job_dir}/kind", capture=True)
        ssh(host, f"echo {args.workdir} > {remote_job_dir}/workdir", capture=True)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            local_config_path = f.name
        try:
            scp(local_config_path, host, remote_config_path)
        finally:
            Path(local_config_path).unlink()

    if kind in _SLURM_KINDS:
        job_id = _submit_slurm(host, args, job_name, remote_job_dir, remote_config_path)
        print(f"  submitted Slurm job {job_id}")
        log_ident = job_id
    elif kind in _TORCHRUN_KINDS or kind in _SINGLE_PROCESS_KINDS:
        _submit_background(host, args, kind, job_name, remote_config_path)
        print("  launched in background (nohup)")
        log_ident = job_name
    else:
        raise SystemExit(f"unsupported cluster kind: {kind!r}")

    if not args.dry_run:
        print(f"  config: {host}:{remote_config_path}")
        print(f"  logs:   {host}:{args.workdir}/logs/{log_ident}-out.log (and -err.log)")


def _cmd_status(args) -> None:
    cluster = get_cluster(args.clusters, args.cluster)
    host = cluster["ssh_host"]
    remote_job_dir = f"{args.jobs_dir}/{args.name}"
    job_kind = ssh(host, f"cat {remote_job_dir}/kind 2>/dev/null", capture=True).stdout.strip() or "train"

    if job_kind == "sweep":
        driver = ssh(host, f"pgrep -af {shlex.quote(remote_job_dir)}", capture=True).stdout.strip()
        queue = ssh(host, f"{slurm_bin('squeue')} -u $(whoami) -h -o '%i|%j|%T|%M|%N'", capture=True).stdout.strip()
        print(f"sweep driver: {driver or 'not running (finished, or --count reached)'}")
        print(f"your queued/running Slurm jobs:\n{queue or '(none)'}")
    elif cluster["kind"] in _SLURM_KINDS:
        job_id = ssh(host, f"cat {remote_job_dir}/jobid 2>/dev/null", capture=True).stdout.strip()
        if not job_id:
            print("no Slurm job id recorded for this job")
            return
        result = ssh(host, f"{slurm_bin('squeue')} -j {job_id} -h -o '%i|%T|%M|%N'", capture=True)
        print(result.stdout.strip() or f"job {job_id} not in queue (finished or failed)")
    else:
        remote_config_path = f"{remote_job_dir}/config.yaml"
        result = ssh(host, f"pgrep -af {shlex.quote(remote_config_path)}", capture=True)
        print(result.stdout.strip() or "not running")


def _resolve_log_path(host: str, cluster: dict, remote_job_dir: str, job_name: str, stream: str) -> str:
    job_kind = ssh(host, f"cat {remote_job_dir}/kind 2>/dev/null", capture=True).stdout.strip() or "train"
    if job_kind == "sweep":
        return f"{remote_job_dir}/run.log"

    workdir = ssh(host, f"cat {remote_job_dir}/workdir 2>/dev/null", capture=True).stdout.strip() or DEFAULT_WORKDIR
    if cluster["kind"] in _SLURM_KINDS:
        # the jobid file is only written after a successful sbatch submission
        ident = ssh(host, f"cat {remote_job_dir}/jobid 2>/dev/null", capture=True).stdout.strip() or job_name
    else:
        ident = job_name
    return f"{workdir}/logs/{ident}-{stream}.log"


def _cmd_logs(args) -> None:
    cluster = get_cluster(args.clusters, args.cluster)
    host = cluster["ssh_host"]
    remote_job_dir = f"{args.jobs_dir}/{args.name}"
    stream = "err" if args.stderr else "out"
    remote_log_path = _resolve_log_path(host, cluster, remote_job_dir, args.name, stream)
    tail_cmd = f"tail -n {args.lines}" + (" -f" if args.follow else "")
    subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, f"{tail_cmd} {remote_log_path}"])


def _cmd_cancel(args) -> None:
    cluster = get_cluster(args.clusters, args.cluster)
    host = cluster["ssh_host"]
    remote_job_dir = f"{args.jobs_dir}/{args.name}"
    job_kind = ssh(host, f"cat {remote_job_dir}/kind 2>/dev/null", capture=True).stdout.strip() or "train"

    if job_kind == "sweep":
        ssh(host, f"pkill -f {shlex.quote(remote_job_dir)}", capture=True)
        print("stopped the sweep driver (no new agent jobs will be submitted)")
        print("any already-running Slurm agent jobs for this sweep keep running — scancel them individually if needed")
    elif cluster["kind"] in _SLURM_KINDS:
        job_id = ssh(host, f"cat {remote_job_dir}/jobid 2>/dev/null", capture=True).stdout.strip()
        if not job_id:
            print("no Slurm job id recorded for this job")
            return
        ssh(host, f"{slurm_bin('scancel')} {job_id}", capture=True)
        print(f"cancelled Slurm job {job_id}")
    else:
        remote_config_path = f"{remote_job_dir}/config.yaml"
        ssh(host, f"pkill -f {shlex.quote(remote_config_path)}", capture=True)
        print(f"sent SIGTERM to processes running {remote_config_path}")


def _cmd_list_clusters(args) -> None:
    clusters = load_clusters(args.clusters)
    for cluster in clusters.values():
        print(f"{cluster['id']:<14} {cluster['kind']:<12} ssh_host={cluster['ssh_host']:<14} {cluster['label']}")


def _cmd_dashboard(args) -> None:
    from .dashboard.server import run_dashboard

    run_dashboard(clusters_path=args.clusters, port=args.port)


def _add_common_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clusters", default=str(DEFAULT_CLUSTERS_YAML), help="path to clusters.yaml")
    parser.add_argument(
        "--cluster",
        required=True,
        help="cluster id from clusters.yaml, or 'auto' to pick whichever has free capacity now",
    )
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
    p_submit.add_argument(
        "--workdir",
        default=None,
        help=f"remote lm-engine checkout to run from (default: the cluster's own 'workdir' in clusters.yaml, else {DEFAULT_WORKDIR!r})",
    )
    p_submit.add_argument(
        "--nodes", type=int, default=1, help="[slurm_gpu] nodes to request (capped by the cluster's max_nodes)"
    )
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
    p_logs.add_argument("--stderr", action="store_true", help="show the error log instead of the output log")
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
