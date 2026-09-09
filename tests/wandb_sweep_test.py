# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""
Smoke tests for tools/wandb_sweep.py: verify that it can actually schedule
jobs through each of its integration points, without touching the real W&B
API, a real Slurm cluster, or the network.

    * local create mode: wandb.sweep() and sbatch are mocked
    * --cluster mode (via lm_engine.virtual_cluster): ssh/scp are mocked
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lm_engine.virtual_cluster import remote as remote_module


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_wandb_sweep():
    spec = importlib.util.spec_from_file_location("wandb_sweep", _REPO_ROOT / "tools" / "wandb_sweep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["wandb_sweep"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wandb_sweep():
    return _load_wandb_sweep()


@pytest.fixture
def base_config_path(tmp_path) -> Path:
    config = {
        "save_args": {"save_path": "checkpoints", "save_interval": 10},
        "logging_args": {"wandb_args": {"project": "test-project", "entity": "test-entity"}},
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(config))
    return path


@pytest.fixture
def sweep_config_path(tmp_path) -> Path:
    sweep = {"method": "grid", "parameters": {"optimizer_args.class_args.lr": {"values": [1e-4, 1e-3]}}}
    path = tmp_path / "sweep.yml"
    path.write_text(yaml.dump(sweep))
    return path


@pytest.fixture
def clusters_yaml_path(tmp_path) -> Path:
    clusters = {
        "clusters": [
            {"id": "testcluster", "label": "Test Cluster", "kind": "slurm_gpu", "ssh_host": "testhost", "max_nodes": 4}
        ]
    }
    path = tmp_path / "clusters.yaml"
    path.write_text(yaml.dump(clusters))
    return path


def test_local_create_mode_schedules_jobs_via_wandb(
    wandb_sweep, monkeypatch, tmp_path, base_config_path, sweep_config_path
) -> None:
    sweep_calls = []
    monkeypatch.setattr(
        wandb_sweep.wandb, "sweep", lambda config, project=None, entity=None: sweep_calls.append(config) or "sweep123"
    )

    sbatch_calls = []

    def fake_run(cmd, capture_output=False, text=False, **kwargs):
        binary = Path(cmd[0]).name
        if binary == "sbatch":
            sbatch_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="Submitted batch job 42\n", stderr="")
        if binary == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected local command in create mode: {cmd}")

    monkeypatch.setattr(wandb_sweep.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_sweep.py",
            "--config",
            str(base_config_path),
            "--sweep",
            str(sweep_config_path),
            "--slurm_logs_dir",
            str(tmp_path / "logs"),
            "--count",
            "2",
            "--max_concurrent",
            "2",
        ],
    )

    wandb_sweep.main()

    assert len(sweep_calls) == 1, "wandb.sweep() should be called exactly once to create the sweep"
    assert sweep_calls[0]["parameters"] == {"optimizer_args.class_args.lr": {"values": [1e-4, 1e-3]}}
    assert len(sbatch_calls) == 2, "one Slurm job should be submitted per --count"
    for cmd in sbatch_calls:
        wrap = cmd[cmd.index("--wrap") + 1]
        assert "--agent" in wrap and "--sweep_id sweep123" in wrap
        assert "--project test-project" in wrap and "--entity test-entity" in wrap


def test_remote_cluster_mode_ships_and_launches_sweep(
    wandb_sweep, monkeypatch, base_config_path, sweep_config_path, clusters_yaml_path
) -> None:
    ssh_calls = []
    scp_calls = []

    def fake_ssh(host, command, capture=False):
        ssh_calls.append((host, command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_scp(local_path, host, remote_path):
        scp_calls.append((local_path, host, remote_path))

    # ensure_job_dir() (called by _run_on_remote_cluster) calls remote.ssh directly, not
    # wandb_sweep's own `from ... import ssh` binding, so both need patching to stay hermetic.
    monkeypatch.setattr(wandb_sweep, "ssh", fake_ssh)
    monkeypatch.setattr(wandb_sweep, "scp", fake_scp)
    monkeypatch.setattr(remote_module, "ssh", fake_ssh)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_sweep.py",
            "--cluster",
            "testcluster",
            "--clusters",
            str(clusters_yaml_path),
            "--config",
            str(base_config_path),
            "--sweep",
            str(sweep_config_path),
            "--count",
            "5",
            "--num_nodes",
            "2",
        ],
    )

    wandb_sweep.main()

    assert len(scp_calls) == 3, "should upload the script + base config + sweep config"
    assert {call[1] for call in scp_calls} == {"testhost"}

    ssh_hosts = {call[0] for call in ssh_calls}
    assert ssh_hosts == {"testhost"}
    commands = [call[1] for call in ssh_calls]
    mkdir_cmd = next(cmd for cmd in commands if cmd.startswith("mkdir -p"))
    assert " && ln -sfn " in mkdir_cmd, "should symlink <jobs_dir>/<job_name> at the real hash-suffixed dir"
    assert any("echo sweep >" in cmd and cmd.endswith("/kind") for cmd in commands)

    launch_cmd = next(cmd for cmd in commands if "nohup" in cmd)
    assert "wandb_sweep.py" in launch_cmd
    assert "--count 5" in launch_cmd
    assert "--num_nodes 2" in launch_cmd
    # the real job dir is hash-suffixed, not exactly ~/lm-engine-jobs/<job-name>
    m = re.search(r"~/lm-engine-jobs/(sweep-config-\S+?)-([0-9a-f]{8})/wandb_sweep\.py", launch_cmd)
    assert m, f"expected a hash-suffixed job dir in: {launch_cmd}"
    # the remote re-invocation must run in plain local create mode, not recurse into --cluster
    assert "--cluster" not in launch_cmd


def test_remote_cluster_mode_rejects_excess_nodes(
    wandb_sweep, monkeypatch, base_config_path, sweep_config_path, clusters_yaml_path
) -> None:
    monkeypatch.setattr(wandb_sweep, "ssh", lambda *a, **k: pytest.fail("ssh should not be called"))
    monkeypatch.setattr(wandb_sweep, "scp", lambda *a, **k: pytest.fail("scp should not be called"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_sweep.py",
            "--cluster",
            "testcluster",
            "--clusters",
            str(clusters_yaml_path),
            "--config",
            str(base_config_path),
            "--sweep",
            str(sweep_config_path),
            "--num_nodes",
            "99",
        ],
    )

    with pytest.raises(SystemExit, match="allows at most 4 node"):
        wandb_sweep.main()


def test_remote_cluster_mode_rejects_non_slurm_cluster(
    wandb_sweep, monkeypatch, tmp_path, base_config_path, sweep_config_path
) -> None:
    clusters = {"clusters": [{"id": "bare", "label": "Bare Box", "kind": "nvidia_gpu", "ssh_host": "barehost"}]}
    clusters_path = tmp_path / "clusters.yaml"
    clusters_path.write_text(yaml.dump(clusters))

    monkeypatch.setattr(wandb_sweep, "ssh", lambda *a, **k: pytest.fail("ssh should not be called"))
    monkeypatch.setattr(wandb_sweep, "scp", lambda *a, **k: pytest.fail("scp should not be called"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wandb_sweep.py",
            "--cluster",
            "bare",
            "--clusters",
            str(clusters_path),
            "--config",
            str(base_config_path),
            "--sweep",
            str(sweep_config_path),
        ],
    )

    with pytest.raises(SystemExit, match="slurm_gpu"):
        wandb_sweep.main()
