# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""
Local dashboard for the virtual cluster defined in clusters.yaml.

Polls each host over SSH, parses cluster-specific tooling output (tpu-info
for TPU VMs; sinfo/squeue/nvidia-smi for Slurm+GPU clusters; rocm-smi for AMD
boxes), and serves a small auto-refreshing HTML page on localhost only.

Usage:
    python -m lm_engine.virtual_cluster dashboard [--port 8765] [--clusters clusters.yaml]
"""

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml


HERE = Path(__file__).parent
DEFAULT_CLUSTERS_YAML = HERE.parent / "clusters.yaml"
DEFAULT_PORT = 8765
STATUS_TIMEOUT = 25
CACHE_TTL = 4  # seconds; avoid hammering SSH if multiple clients poll

VALID_KINDS = {"tpu", "slurm_gpu", "amd_gpu", "nvidia_gpu"}
DEFAULT_SCRIPTS = {
    "tpu": "remote_collect_tpu.sh",
    "slurm_gpu": "remote_collect_slurm_gpu.sh",
    "amd_gpu": "remote_collect_amd_gpu.sh",
    "nvidia_gpu": "remote_collect_nvidia_gpu.sh",
}


def load_clusters(path: Path) -> list:
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        sys.exit(f"cluster config not found: {path}")
    except yaml.YAMLError as e:
        sys.exit(f"failed to parse cluster config {path}: {e}")

    entries = (raw or {}).get("clusters") or []
    clusters = []
    for i, entry in enumerate(entries):
        cid = entry.get("id")
        kind = entry.get("kind")
        if not cid:
            sys.exit(f"cluster config {path}: entry {i} is missing required field 'id'")
        if kind not in VALID_KINDS:
            sys.exit(
                f"cluster config {path}: cluster '{cid}' has invalid kind {kind!r} (expected one of {sorted(VALID_KINDS)})"
            )
        script = entry.get("script") or DEFAULT_SCRIPTS[kind]
        clusters.append(
            {
                "id": cid,
                "label": entry.get("label") or cid,
                "kind": kind,
                "ssh_host": entry.get("ssh_host") or cid,
                "script": HERE / script,
            }
        )
    return clusters


_CLUSTERS: list = []
_cache = {"ts": 0, "data": None}


# ---------------------------------------------------------------- transport


def run_remote(ssh_host: str, script_path: Path) -> dict:
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "BatchMode=yes",
                "-o",
                "ExitOnForwardFailure=no",
                ssh_host,
                "bash",
                "-s",
            ],
            input=script_path.read_text(),
            capture_output=True,
            text=True,
            timeout=STATUS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"SSH to {ssh_host} timed out after {STATUS_TIMEOUT}s"}

    if "@@SECTION:HOST@@" not in proc.stdout:
        err = proc.stderr.strip() or "no output"
        return {"ok": False, "error": f"SSH to {ssh_host} failed: {err}"}

    return {"ok": True, "raw": proc.stdout}


def split_sections(raw: str) -> dict:
    parts = re.split(r"@@SECTION:(\w+)@@\n", raw)
    sections = {}
    for i in range(1, len(parts), 2):
        sections[parts[i]] = parts[i + 1]
    return sections


# ------------------------------------------------------------- common bits


def parse_common(sections: dict) -> dict:
    mem = {}
    m = re.search(r"Mem:\s+(\d+)\s+(\d+)\s+(\d+)", sections.get("MEM", ""))
    if m:
        total, used, free = (int(x) for x in m.groups())
        mem = {"total_mb": total, "used_mb": used, "free_mb": free}

    disk = {}
    m = re.search(r"\S+\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)%", sections.get("DISK", ""))
    if m:
        size, used, avail, pct = m.groups()
        disk = {"size": size, "used": used, "avail": avail, "pct": int(pct)}

    load_m = re.search(r"load average:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", sections.get("UPTIME", ""))
    uptime_m = re.search(r"up\s+(.*?),\s+[\d.]+ users?,", sections.get("UPTIME", ""))

    return {
        "host": sections.get("HOST", "").strip(),
        "uptime": uptime_m.group(1) if uptime_m else None,
        "load": [float(x) for x in load_m.groups()] if load_m else None,
        "mem": mem,
        "disk": disk,
    }


def parse_ps(ps_text: str) -> dict:
    """pid -> {user, pcpu, pmem, etimes, cmd}"""
    procs = {}
    for line in ps_text.strip().splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6:
            continue
        pid, user, pcpu, pmem, etimes, cmd = fields
        try:
            procs[int(pid)] = {
                "user": user,
                "pcpu": float(pcpu),
                "pmem": float(pmem),
                "etimes": int(etimes),
                "cmd": cmd,
            }
        except ValueError:
            continue
    return procs


# -------------------------------------------------------------------- TPU


def parse_tpu_info(text: str) -> dict:
    result = {"accelerator": None, "libtpu_version": None, "chips": [], "note": None}

    m = re.search(r"Accelerator type:\s*(\S+)", text)
    if m:
        result["accelerator"] = m.group(1)
    m = re.search(r"Libtpu version:\s*(\S+)", text)
    if m:
        result["libtpu_version"] = m.group(1)

    if "TPU_INFO_UNAVAILABLE" in text:
        result["note"] = "tpu-info not found on host"
        return result
    if "segfaulted during canary process" in text:
        result["note"] = "libtpu canary segfaulted (known SDK/Python version mismatch) — chip list still read from PCI"

    chips = {}
    for m in re.finditer(r"\|\s*(/dev/vfio/\d+)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|", text):
        path, chip_type, devices, pid = m.groups()
        idx = int(path.rsplit("/", 1)[-1])
        chips[idx] = {
            "index": idx,
            "path": path,
            "type": chip_type.strip(),
            "devices": int(devices),
            "pid": None if pid in ("N/A", "None") else int(pid),
            "hbm_gib": None,
            "hbm_total_gib": None,
            "duty_cycle_pct": None,
        }

    for m in re.finditer(r"\|\s*(\d+)\s*\|\s*([\d.]+|N/A)\s*(?:/\s*([\d.]+))?\s*\|\s*([\d.]+|N/A)\s*%?\s*\|", text):
        idx, used, total, duty = m.groups()
        idx = int(idx)
        if idx not in chips:
            continue
        chips[idx]["hbm_gib"] = None if used == "N/A" else float(used)
        chips[idx]["hbm_total_gib"] = float(total) if total else None
        chips[idx]["duty_cycle_pct"] = None if duty == "N/A" else float(duty)

    result["chips"] = [chips[i] for i in sorted(chips)]
    return result


def build_tpu_status(sections: dict) -> dict:
    tpu = parse_tpu_info(sections.get("TPUINFO", ""))
    procs = parse_ps(sections.get("PS", ""))

    jobs = []
    seen_pids = set()
    for chip in tpu["chips"]:
        pid = chip["pid"]
        if pid and pid not in seen_pids:
            seen_pids.add(pid)
            info = procs.get(pid, {})
            jobs.append(
                {
                    "pid": pid,
                    "user": info.get("user", "?"),
                    "cmd": info.get("cmd", "?"),
                    "pcpu": info.get("pcpu"),
                    "pmem": info.get("pmem"),
                    "etimes": info.get("etimes"),
                    "chips": [c["index"] for c in tpu["chips"] if c["pid"] == pid],
                }
            )

    return {"tpu": tpu, "jobs": jobs}


# -------------------------------------------------------------- Slurm/GPU


def parse_sinfo(text: str) -> list:
    nodes = {}
    order = []
    for line in text.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        name, partition, state, gres = parts
        if name not in nodes:
            nodes[name] = {"name": name, "state": state, "partitions": [], "gpus": 0}
            order.append(name)
        nodes[name]["partitions"].append(partition)
        gm = re.search(r"gpu:(\d+)", gres)
        if gm:
            nodes[name]["gpus"] = int(gm.group(1))
    return [nodes[n] for n in order]


def parse_squeue(text: str) -> list:
    jobs = []
    for line in text.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 8:
            continue
        jobid, partition, name, user, state, elapsed, nnodes, nodelist = parts
        jobs.append(
            {
                "jobid": jobid,
                "partition": partition,
                "name": name,
                "user": user,
                "state": state,
                "elapsed": elapsed,
                "nnodes": nnodes,
                "nodelist": nodelist,
            }
        )
    return jobs


def parse_mygpu(text: str) -> dict:
    result = {}
    current = None
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        hdr = re.match(r"###\s*(\S+)", line)
        if hdr:
            current = hdr.group(1)
            result[current] = []
            continue
        if current is None:
            continue
        m = re.match(r"(\d+),\s*(.+?),\s*(\d+)\s*%,\s*(\d+)\s*MiB,\s*(\d+)\s*MiB,\s*(\d+)", line)
        if m:
            idx, name, util, mem_used, mem_total, temp = m.groups()
            result[current].append(
                {
                    "index": int(idx),
                    "name": name.strip(),
                    "util_pct": int(util),
                    "mem_used_mib": int(mem_used),
                    "mem_total_mib": int(mem_total),
                    "temp_c": int(temp),
                }
            )
        # else: srun/timeout error text for this job — silently skipped
    return result


def build_slurm_gpu_status(sections: dict) -> dict:
    username = sections.get("WHOAMI", "").strip()
    nodes = parse_sinfo(sections.get("SINFO", ""))
    queue = parse_squeue(sections.get("SQUEUE", ""))
    my_gpu = parse_mygpu(sections.get("MYGPU", ""))
    my_jobs = [j for j in queue if j["user"] == username] if username else []
    return {
        "username": username or None,
        "nodes": nodes,
        "queue": queue,
        "my_jobs": my_jobs,
        "my_gpu": my_gpu,
    }


# --------------------------------------------------------------- AMD/ROCm


def parse_rocm_smi(text: str) -> list:
    gpus = []
    row_re = re.compile(r"^(\d+)\s+\d+\s+\S+\s+\S+\s+([\d.]+)°C\s+([\d.]+)W\s+.*\s(\d+)%\s+(\d+)%\s*$")
    for line in text.splitlines():
        m = row_re.match(line.rstrip())
        if not m:
            continue
        idx, temp, power, vram_pct, gpu_pct = m.groups()
        gpus.append(
            {
                "index": int(idx),
                "temp_c": float(temp),
                "power_w": float(power),
                "vram_pct": int(vram_pct),
                "gpu_pct": int(gpu_pct),
            }
        )
    return gpus


def parse_rocm_pids(text: str) -> list:
    procs = []
    for line in text.strip().splitlines():
        fields = line.split()
        if len(fields) != 6 or not fields[0].isdigit():
            continue
        pid, name, gpu_list, vram_used, sdma_used, cu_occ = fields
        procs.append(
            {
                "pid": int(pid),
                "name": name,
                "gpus": [int(x) for x in gpu_list.split(",") if x.strip().isdigit()],
                "vram_used_bytes": int(vram_used) if vram_used.isdigit() else 0,
                "cu_occupancy": cu_occ,
            }
        )
    return procs


def build_amd_gpu_status(sections: dict) -> dict:
    gpus = parse_rocm_smi(sections.get("GPUINFO", ""))
    gpu_procs = parse_rocm_pids(sections.get("GPUPIDS", ""))
    procs = parse_ps(sections.get("PS", ""))

    jobs = []
    for gp in gpu_procs:
        info = procs.get(gp["pid"], {})
        jobs.append(
            {
                "pid": gp["pid"],
                "name": gp["name"],
                "user": info.get("user", "?"),
                "cmd": info.get("cmd", gp["name"]),
                "gpus": gp["gpus"],
                "vram_used_gib": gp["vram_used_bytes"] / (1024**3),
                "pcpu": info.get("pcpu"),
                "pmem": info.get("pmem"),
                "etimes": info.get("etimes"),
            }
        )

    return {"gpus": gpus, "jobs": jobs}


# ------------------------------------------------------------------ NVIDIA


def parse_nvidia_smi(text: str) -> list:
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        idx, uuid, name, util, mem_used, mem_total, temp = parts
        try:
            gpus.append(
                {
                    "index": int(idx),
                    "uuid": uuid,
                    "name": name,
                    "gpu_pct": int(util.rstrip(" %")),
                    "mem_used_mib": int(mem_used.rstrip(" MiB")),
                    "mem_total_mib": int(mem_total.rstrip(" MiB")),
                    "temp_c": float(temp),
                }
            )
        except ValueError:
            continue
    return gpus


def parse_nvidia_procs(text: str) -> list:
    procs = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        pid, name, used_mem, uuid = parts
        if not pid.isdigit():
            continue
        try:
            mem_mib = int(used_mem.rstrip(" MiB"))
        except ValueError:
            mem_mib = 0
        procs.append({"pid": int(pid), "name": name, "mem_used_mib": mem_mib, "uuid": uuid})
    return procs


def build_nvidia_gpu_status(sections: dict) -> dict:
    gpuinfo_text = sections.get("GPUINFO", "")
    gpus = parse_nvidia_smi(gpuinfo_text)

    note = None
    if "couldn't communicate with the NVIDIA driver" in gpuinfo_text:
        note = "nvidia-smi can't reach the NVIDIA driver (GPU hardware present on PCI bus, but the nvidia kernel module isn't loaded) — needs admin attention on the host"
    elif not gpus and gpuinfo_text.strip():
        note = "nvidia-smi returned unexpected output: " + gpuinfo_text.strip().splitlines()[0][:200]

    uuid_to_index = {g["uuid"]: g["index"] for g in gpus}
    gpu_procs = parse_nvidia_procs(sections.get("GPUPROCS", ""))
    procs = parse_ps(sections.get("PS", ""))

    jobs = []
    for gp in gpu_procs:
        info = procs.get(gp["pid"], {})
        jobs.append(
            {
                "pid": gp["pid"],
                "name": gp["name"],
                "user": info.get("user", "?"),
                "cmd": info.get("cmd", gp["name"]),
                "gpu_index": uuid_to_index.get(gp["uuid"]),
                "vram_used_gib": gp["mem_used_mib"] / 1024,
                "pcpu": info.get("pcpu"),
                "pmem": info.get("pmem"),
                "etimes": info.get("etimes"),
            }
        )

    return {
        "gpus": [{k: v for k, v in g.items() if k != "uuid"} for g in gpus],
        "jobs": jobs,
        "note": note,
    }


# ------------------------------------------------------------- orchestrate


def build_cluster_status(cluster: dict) -> dict:
    remote = run_remote(cluster["ssh_host"], cluster["script"])
    base = {"id": cluster["id"], "label": cluster["label"], "kind": cluster["kind"]}
    if not remote["ok"]:
        return {**base, "ok": False, "error": remote["error"]}

    sections = split_sections(remote["raw"])
    common = parse_common(sections)

    if cluster["kind"] == "tpu":
        specific = build_tpu_status(sections)
    elif cluster["kind"] == "amd_gpu":
        specific = build_amd_gpu_status(sections)
    elif cluster["kind"] == "nvidia_gpu":
        specific = build_nvidia_gpu_status(sections)
    else:
        specific = build_slurm_gpu_status(sections)

    return {**base, "ok": True, **common, **specific}


def build_status() -> dict:
    with ThreadPoolExecutor(max_workers=max(len(_CLUSTERS), 1)) as pool:
        results = list(pool.map(build_cluster_status, _CLUSTERS))
    return {"fetched_at": time.time(), "clusters": results}


def get_status_cached() -> dict:
    now = time.time()
    if _cache["data"] and now - _cache["ts"] < CACHE_TTL:
        return _cache["data"]
    data = build_status()
    _cache["data"] = data
    _cache["ts"] = now
    return data


INDEX_HTML = (HERE / "index.html").read_text()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            data = get_status_cached()
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def run_dashboard(clusters_path: str | Path | None = None, port: int = DEFAULT_PORT) -> None:
    global _CLUSTERS

    path = Path(clusters_path) if clusters_path else DEFAULT_CLUSTERS_YAML
    _CLUSTERS = load_clusters(path)
    _cache["data"] = None
    _cache["ts"] = 0

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Cluster dashboard: http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    run_dashboard(os.environ.get("CLUSTERS_CONFIG"), int(os.environ.get("PORT", DEFAULT_PORT)))


if __name__ == "__main__":
    main()
