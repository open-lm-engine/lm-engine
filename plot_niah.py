# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

#!/usr/bin/env python3
"""
Plot NIAH results from logs-400m-niah.
Supports: (1) lm-eval JSON results in subdirs results-<model>-400m-cosine,
          (2) flat .txt files with lines "Model;  score1 score2 ..."
Edit the variables at the top to choose which runs/models to ignore.
"""

import fnmatch
import json
import math
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


matplotlib.rcParams.update({"font.size": 16})
n = 3


# -----------------------------------------------------------------------------
# Config: edit these (no need to pass args)
# -----------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs-400m-niah"
IGNORE = []  # Result dir or file name patterns to ignore (glob or substring)
IGNORE_MODELS = [
    "RNN",
    "GRU",
    "Gated DeltaNet",
    "Hybrid Gated DeltaNet",
    "Hybrid Gated DeltaNet + $\mathrm{M}^2\mathrm{RNN}$-1",
    "Hybrid Gated DeltaNet + $\mathrm{M}^2\mathrm{RNN}$-" + f"{n}",
    "Gated DeltaNet (neg)",
    "GDN",
]  # Model names to exclude from plot, e.g. ["rnn", "gru"]
# IGNORE_MODELS = [
#     "RNN",
#     "GRU",
#     "Hybrid Mamba2",
#     "Hybrid Mamba2 + $\mathrm{M}^2\mathrm{RNN}$-1",
#     "Hybrid Mamba2 + $\mathrm{M}^2\mathrm{RNN}$-" + f"{n}",
#     "Gated DeltaNet (neg)",
#     "Mamba2",
# ]  # Model names to exclude from plot, e.g. ["rnn", "gru"]
OUTPUT = Path("niah.svg")
NCOLS = 3
FIGSIZE_PER_ROW = (18, 5)
# For JSON mode: model size in dir name (e.g. "400m") and display names
SIZE = "400m"
MODEL_NAME_MAP = {
    "softmax-attention": "Softmax Attention",
    "mamba2": "Mamba2",
    "gated-deltanet": "Gated DeltaNet",
    "gated-deltanet-neg": "Gated DeltaNet (neg)",
    "gru": "GRU",
    "rnn": "RNN",
    "rsa": "$\mathrm{M}^2\mathrm{RNN}$",
    "hybrid-mamba2": "Hybrid Mamba2",
    "hybrid-mamba2-rsa-1l": "Hybrid Mamba2 + $\mathrm{M}^2\mathrm{RNN}$-1",
    "hybrid-mamba2-rsa-nl": "Hybrid Mamba2 + $\mathrm{M}^2\mathrm{RNN}$-" + f"{n}",
    "hybrid-gated-deltanet": "Hybrid Gated DeltaNet",
    "hybrid-gated-deltanet-rsa-1l": "Hybrid Gated DeltaNet + $\mathrm{M}^2\mathrm{RNN}$-1",
    "hybrid-gated-deltanet-rsa-nl": "Hybrid Gated DeltaNet + $\mathrm{M}^2\mathrm{RNN}$-" + f"{n}",
    "hybrid-rsa": "Hybrid $\mathrm{M}^2\mathrm{RNN}$",
}
# Canonical display order and pinned colours (consistent across all subplots)
MODEL_ORDER = [
    "Softmax Attention",
    "Mamba2",
    "Gated DeltaNet",
    "Gated DeltaNet (neg)",
    "GRU",
    "RNN",
    "$\\mathrm{M}^2\\mathrm{RNN}$",
    "Hybrid Mamba2",
    "Hybrid Mamba2 + $\\mathrm{M}^2\\mathrm{RNN}$-1",
    "Hybrid Mamba2 + $\\mathrm{M}^2\\mathrm{RNN}$-" + f"{n}",
    "Hybrid Gated DeltaNet",
    "Hybrid Gated DeltaNet + $\\mathrm{M}^2\\mathrm{RNN}$-1",
    "Hybrid Gated DeltaNet + $\\mathrm{M}^2\\mathrm{RNN}$-" + f"{n}",
    "Hybrid $\\mathrm{M}^2\\mathrm{RNN}$",
]
_tab10 = plt.cm.tab10.colors
MODEL_COLORS = {name: _tab10[i % len(_tab10)] for i, name in enumerate(MODEL_ORDER)}

MODEL_COLORS["Hybrid Gated DeltaNet"] = (0.1, 0.1, 0.1)
MODEL_COLORS["Gated DeltaNet"] = (0.5, 0.5, 0.5)

NIAH_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multiquery",
    "niah_multikey_1",
    "niah_multivalue",
]
TASK_TITLES = ["S-NIAH-1", "S-NIAH-2", "S-NIAH-3", "MQ-NIAH", "MK-NIAH", "MV-NIAH"]
# -----------------------------------------------------------------------------

SEQUENCE_LENGTHS = [1024, 2048, 4096, 8192, 16384]


def should_ignore(name: str, ignore_patterns: list[str]) -> bool:
    """True if name matches any ignore pattern (glob or substring)."""
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(name, pattern) or pattern in name:
            return True
    return False


def find_json_results(log_dir: Path, size: str) -> dict[str, Path]:
    """Find lm-eval JSON files in results-<name>-<size>-cosine subdirs."""
    out = {}
    pattern = re.compile(re.escape(size).join([r"results-(.+)-", r"-cosine"]))
    for subdir in log_dir.iterdir():
        if not subdir.is_dir() or not subdir.name.startswith("results-"):
            continue
        match = pattern.match(subdir.name)
        if not match:
            continue
        model_name = match.group(1)
        if should_ignore(subdir.name, IGNORE):
            continue
        jsons = list(subdir.rglob("*.json"))
        if jsons:
            out[model_name] = jsons[0]
    return out


def load_niah_from_jsons(log_dir: Path, size: str) -> list[tuple[dict[str, list[float]], str]]:
    """Load NIAH metrics from lm-eval JSONs. Returns list of (model -> scores, title) per task."""
    model_files = find_json_results(log_dir, size)
    if not model_files:
        return []
    ignore_models = {m.strip().lower() for m in IGNORE_MODELS}
    task_data = []
    for task, title in zip(NIAH_TASKS, TASK_TITLES):
        results = {}
        for model_name, path in model_files.items():
            if model_name.lower() in ignore_models:
                continue
            with open(path) as f:
                data = json.load(f)
            res = data.get("results", {}).get(task, {})
            scores = []
            for L in SEQUENCE_LENGTHS:
                key = f"{L},none"
                scores.append(res.get(key, 0.0) * 100.0 if isinstance(res.get(key), (int, float)) else 0.0)
            display = MODEL_NAME_MAP.get(model_name, model_name.replace("-", " ").title())
            results[display] = scores
        if results:
            task_data.append((results, title))
    return task_data


def parse_task_file(path: Path) -> tuple[dict[str, list[float]], str]:
    """Parse a .txt task file. Returns (model -> list of scores), title (from filename)."""
    text = path.read_text()
    results = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ";" not in line:
            continue
        model, rest = line.split(";", 1)
        model = model.strip()
        parts = rest.split()
        scores = [float(x) for x in parts if x]
        if scores:
            results[model] = scores
    title = path.stem.replace("_", " ").replace("-", " ")
    return results, title


def load_niah_from_txt(log_dir: Path, ext: str) -> list[tuple[dict[str, list[float]], str]]:
    """Load task data from flat .txt files (one file per subplot)."""
    files = sorted(log_dir.glob(f"*{ext}"))
    if IGNORE:
        files = [f for f in files if not should_ignore(f.name, IGNORE)]
    task_data = []
    for path in files:
        results, title = parse_task_file(path)
        if results:
            task_data.append((results, title))
    return task_data


def plot_task(results: dict[str, list[float]], sequence_lengths: list[int], ax, title: str, ignore_models: set[str]):
    """Draw one subplot for one task."""
    for model in MODEL_ORDER:
        if model not in results or model in ignore_models:
            continue
        scores = results[model]
        if len(scores) != len(sequence_lengths):
            continue
        ax.plot(sequence_lengths, scores, marker="o", linewidth=2, label=model, color=MODEL_COLORS.get(model))
    for model, scores in results.items():
        if model in ignore_models or model in MODEL_COLORS:
            continue
        if len(scores) != len(sequence_lengths):
            continue
        ax.plot(sequence_lengths, scores, marker="o", linewidth=2, label=model)
    ax.set_xticks(sequence_lengths)
    ax.set_xlabel("Sequence Length")
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(sequence_lengths, rotation=45, ha="right")
    ax.grid(True)
    ax.axvline(x=4096, color="black", linestyle="--", linewidth=1.5, alpha=0.8)


def main():
    if not LOG_DIR.is_dir():
        raise SystemExit(f"Log directory not found: {LOG_DIR}")

    # Prefer lm-eval JSON results in subdirs; fall back to .txt files
    task_data = load_niah_from_jsons(LOG_DIR, SIZE)
    if not task_data:
        task_data = load_niah_from_txt(LOG_DIR, ".txt")
    if not task_data:
        raise SystemExit(
            f"No NIAH data found. Either add result subdirs results-<model>-{SIZE}-cosine with JSON, "
            "or add .txt task files."
        )

    ignore_models = set(m.strip() for m in IGNORE_MODELS)

    n_tasks = len(task_data)
    n_rows = math.ceil(n_tasks / NCOLS)
    fig_w, fig_h_per_row = FIGSIZE_PER_ROW
    fig, axes = plt.subplots(n_rows, NCOLS, figsize=(fig_w, fig_h_per_row * n_rows), sharey=True)
    sns.set(style="whitegrid")

    if n_rows == 1 and NCOLS == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif NCOLS == 1:
        axes = [[ax] for ax in axes]

    for idx, (results, title) in enumerate(task_data):
        r, c = divmod(idx, NCOLS)
        plot_task(results, SEQUENCE_LENGTHS, axes[r][c], title, ignore_models)

    for idx in range(n_tasks, n_rows * NCOLS):
        r, c = divmod(idx, NCOLS)
        axes[r][c].set_visible(False)

    for r in range(n_rows):
        axes[r][0].set_ylabel("Accuracy (%)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False, fontsize=16)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(OUTPUT, format=OUTPUT.suffix.lstrip(".") or "svg")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
