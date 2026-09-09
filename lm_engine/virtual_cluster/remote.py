# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""
Shared primitives for talking to the virtual cluster defined in
clusters.yaml: resolving a cluster id (including 'auto', which picks
whichever cluster has free capacity right now), and running commands /
copying files onto it over SSH.

Used by both `lm_engine.virtual_cluster.schedule` (submit/status/logs/cancel
training jobs) and `tools/wandb_sweep.py` (--cluster flag, to drive a W&B
sweep on a remote Slurm cluster from your own machine instead of an
interactive SSH session).
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


DEFAULT_CLUSTERS_YAML = Path(__file__).parent / "clusters.yaml"
DEFAULT_WORKDIR = "~/lm-engine"
DEFAULT_JOBS_DIR = "~/lm-engine-jobs"


def load_clusters(path: str) -> dict[str, dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    clusters = {}
    for entry in data["clusters"]:
        cid = entry["id"]
        clusters[cid] = {
            "id": cid,
            "label": entry.get("label", cid),
            "kind": entry["kind"],
            "ssh_host": entry.get("ssh_host", cid),
            "max_nodes": entry.get("max_nodes"),
        }
    return clusters


def check_node_limit(cluster: dict, requested_nodes: int) -> None:
    max_nodes = cluster.get("max_nodes")
    if max_nodes is not None and requested_nodes > max_nodes:
        raise SystemExit(f"cluster {cluster['id']!r} allows at most {max_nodes} node(s); requested {requested_nodes}")


def get_cluster(clusters_path: str, cluster_id: str, allowed_kinds: set[str] | None = None) -> dict:
    if cluster_id == "auto":
        return pick_available_cluster(clusters_path, allowed_kinds=allowed_kinds)

    clusters = load_clusters(clusters_path)
    cluster = clusters.get(cluster_id)
    if cluster is None:
        raise SystemExit(f"unknown cluster {cluster_id!r}; known clusters: {', '.join(sorted(clusters))} (or 'auto')")
    if allowed_kinds and cluster["kind"] not in allowed_kinds:
        raise SystemExit(
            f"cluster {cluster_id!r} is kind {cluster['kind']!r}; expected one of {sorted(allowed_kinds)}"
        )
    return cluster


def cluster_is_available(status: dict) -> tuple[bool, str]:
    if not status.get("ok"):
        return False, status.get("error", "unreachable")

    if status["kind"] == "slurm_gpu":
        idle_nodes = [n for n in status.get("nodes", []) if n.get("state") in ("idle", "mix")]
        if idle_nodes:
            return True, f"{len(idle_nodes)} idle/partially-free node(s)"
        return False, "no idle nodes"

    # tpu / nvidia_gpu / amd_gpu: single-tenant boxes -> free if nothing is running on them
    jobs = status.get("jobs", [])
    if jobs:
        return False, f"{len(jobs)} job(s) already running"
    return True, "no jobs running"


def pick_available_cluster(clusters_path: str, allowed_kinds: set[str] | None = None) -> dict:
    # imported lazily: pulls in the dashboard's SSH-probing status logic only when actually picking
    from .dashboard.server import build_cluster_status
    from .dashboard.server import load_clusters as _load_dashboard_clusters

    clusters = _load_dashboard_clusters(Path(clusters_path))
    if allowed_kinds:
        clusters = [c for c in clusters if c["kind"] in allowed_kinds]
    if not clusters:
        raise SystemExit("no clusters of the required kind are defined in clusters.yaml")

    print(f"[auto] checking {len(clusters)} cluster(s) for free capacity...")
    with ThreadPoolExecutor(max_workers=len(clusters)) as pool:
        statuses = list(pool.map(build_cluster_status, clusters))

    for cluster, status in zip(clusters, statuses):
        available, reason = cluster_is_available(status)
        print(f"  {cluster['id']:<14} {'available' if available else 'busy/unreachable':<17} ({reason})")
        if available:
            print(f"[auto] selected {cluster['id']}")
            return {
                "id": cluster["id"],
                "label": cluster["label"],
                "kind": cluster["kind"],
                "ssh_host": cluster["ssh_host"],
                "max_nodes": cluster.get("max_nodes"),
            }

    raise SystemExit("no cluster currently has free capacity; try again later or pick one explicitly with --cluster")


def ssh(host: str, command: str, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, command]
    return subprocess.run(cmd, capture_output=capture, text=True)


def scp(local_path: str, host: str, remote_path: str) -> None:
    result = subprocess.run(
        ["scp", "-o", "ConnectTimeout=10", local_path, f"{host}:{remote_path}"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"scp to {host} failed: {result.stderr.strip()}")
