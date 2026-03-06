# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

#!/usr/bin/env python3
"""
Generate a LaTeX table from lm-eval results in logs-{size}-lm directories.
Supports 7b and 400m model sizes.
"""

import json
import re
from pathlib import Path


# Supported model sizes; each may have a logs-{size}-lm directory
MODEL_SIZES = ("7b", "400m")

# Base path for log directories (logs-7b-lm, logs-400m-lm, ...)
LOGS_BASE = Path(__file__).resolve().parent

# Benchmarks to include in the table (in order)
# Format: (task_name, metric_key, display_name)
BENCHMARKS = [
    ("wikitext", "word_perplexity,none", "Wiki PPL"),
    ("lambada_openai", "perplexity,none", "LMB PPL"),
    ("lambada_openai", "acc,none", "LAMBADA"),
    ("hellaswag", "acc_norm,none", "HellaSwag"),
    ("piqa", "acc,none", "PIQA"),
    ("arc_easy", "acc,none", "ARC-E"),
    ("arc_challenge", "acc_norm,none", "ARC-C"),
    ("winogrande", "acc,none", "WinoGrande"),
    ("boolq", "acc,none", "BoolQ"),
    ("openbookqa", "acc_norm,none", "OBQA"),
    ("copa", "acc,none", "COPA"),
    ("sciq", "acc,none", "SciQ"),
    # ("race", "acc,none", "RACE"),
]

# Average columns configuration
# Format: list of (display_name, [list of task names to average], higher_is_better)
# Set to empty list [] to disable averaging
AVERAGE_COLUMNS = [
    (
        "Avg Acc",
        [
            "lambada_openai",
            "hellaswag",
            "piqa",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "boolq",
            "openbookqa",
            "copa",
            "sciq",
            "race",
        ],
        True,
    ),  # Average of all accuracy benchmarks (higher is better)
]

# Model name mappings for cleaner display
MODEL_NAME_MAP = {
    "softmax-attention": "Softmax Attention",
    "mamba2": "Mamba2",
    "gated-deltanet": "Gated DeltaNet",
    "gated-deltanet-neg": "Gated DeltaNet (neg)",
    "gru": "GRU",
    "rnn": "RNN",
    "rsa": "RSA",
    "hybrid-mamba2": "Hybrid Mamba2",
    "hybrid-mamba2-rsa-1l": "Hybrid Mamba2 + RSA (1L)",
    "hybrid-mamba2-rsa-nl": "Hybrid Mamba2 + RSA (NL)",
    "hybrid-gated-deltanet": "Hybrid Gated DeltaNet",
    "hybrid-gated-deltanet-rsa-1l": "Hybrid Gated DeltaNet + RSA (1L)",
    "hybrid-gated-deltanet-rsa-nl": "Hybrid Gated DeltaNet + RSA (NL)",
    "hybrid-rsa": "Hybrid RSA",
}


def extract_model_name(dir_name: str, size: str) -> str:
    """Extract model name from directory name like 'results-mamba2-7b-cosine' or 'results-mamba2-400m-cosine'."""
    # Remove 'results-' prefix and '-{size}-cosine' suffix
    pattern = re.escape(size).join([r"results-(.+)-", r"-cosine"])
    match = re.match(pattern, dir_name)
    if match:
        return match.group(1)
    return dir_name


def find_json_files(logs_dir: Path, size: str) -> dict[str, Path]:
    """Find all JSON result files and map them to model names."""
    model_files = {}
    for result_dir in logs_dir.iterdir():
        if not result_dir.is_dir() or not result_dir.name.startswith("results-"):
            continue
        model_name = extract_model_name(result_dir.name, size)
        json_files = list(result_dir.rglob("*.json"))
        if json_files:
            model_files[model_name] = json_files[0]
    return model_files


def load_results(json_path: Path) -> dict:
    """Load results from a JSON file."""
    with open(json_path) as f:
        return json.load(f)


def format_value(value: float, metric_key: str) -> str:
    """Format a metric value for display."""
    if "perplexity" in metric_key:
        # Perplexity: show 2 decimal places
        return f"{value:.2f}"
    else:
        # Accuracy: show as percentage with 1 decimal place
        return f"{value * 100:.2f}"


def compute_average(model_results: dict, tasks: list[str], benchmarks: list) -> float | None:
    """Compute average of specified tasks for a model."""
    # Build task -> metric_key mapping from BENCHMARKS
    task_to_metric = {task: metric_key for task, metric_key, _ in benchmarks}

    values = []
    results = model_results.get("results", {})
    for task in tasks:
        if task not in task_to_metric:
            continue
        metric_key = task_to_metric[task]
        if task in results and metric_key in results[task]:
            val = results[task][metric_key]
            if isinstance(val, (int, float)):
                # Convert to percentage for accuracy metrics
                if "perplexity" not in metric_key:
                    val = val * 100
                values.append(val)

    if values:
        return sum(values) / len(values)
    return None


def _size_display(size: str) -> str:
    """Display string for model size (e.g. 7b -> 7B, 400m -> 400M)."""
    if size == "7b":
        return "7B"
    if size == "400m":
        return "400M"
    return size.upper()


def generate_latex_table(model_results: dict[str, dict], size: str = "7b") -> str:
    """Generate LaTeX table from model results."""
    size_display = _size_display(size)

    # Sort models for consistent ordering; groups separated by midrule:
    # Group 0: base SSMs | Group 1: Softmax + hybrids | Group 2: Mamba2 1L/NL | Group 3: Gated DeltaNet 1L/NL
    model_order = [
        "mamba2",
        "gated-deltanet",
        "gated-deltanet-neg",
        "rnn",
        "gru",
        "rsa",
        "softmax-attention",
        "hybrid-mamba2",
        "hybrid-gated-deltanet",
        "hybrid-rsa",
        "hybrid-mamba2-rsa-1l",
        "hybrid-mamba2-rsa-nl",
        "hybrid-gated-deltanet-rsa-1l",
        "hybrid-gated-deltanet-rsa-nl",
    ]

    # Filter to only include models we have results for
    models = [m for m in model_order if m in model_results]
    # Add any models not in the predefined order (append after their group)
    for m in model_results:
        if m not in models:
            models.append(m)

    # Build header
    header_cols = ["Model"] + [b[2] for b in BENCHMARKS]
    # Add average column headers
    for avg_name, _, _ in AVERAGE_COLUMNS:
        header_cols.append(avg_name)

    num_cols = len(header_cols)
    col_spec = "l" + "c" * (num_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{size_display} Model Evaluation Results}}",
        f"\\label{{tab:{size}-results}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        " & ".join(header_cols) + r" \\",
        r"\midrule",
    ]

    # Assign each model to a group for midrules and per-group bolding:
    # 0 = base SSMs, 1 = Softmax + hybrids (no 1L/NL), 2 = Mamba2 1L/NL, 3 = Gated DeltaNet 1L/NL
    def model_group(model: str) -> int:
        if model == "softmax-attention":
            return 1
        if not model.startswith("hybrid-"):
            return 0
        if "hybrid-mamba2-rsa-" in model:
            return 2
        if "hybrid-gated-deltanet-rsa-" in model:
            return 3
        return 1

    # Find best values per group (for bolding within each midrule section)
    best_values_by_group = {}  # (group, task, metric_key) -> best value
    for group in (0, 1, 2, 3):
        group_models = [m for m in models if model_group(m) == group]
        for task, metric_key, _ in BENCHMARKS:
            if "perplexity" not in metric_key:
                continue
            values = []
            for model in group_models:
                results = model_results[model].get("results", {})
                if task in results and metric_key in results[task]:
                    val = results[task][metric_key]
                    if isinstance(val, (int, float)):
                        values.append(val)
            if values:
                best_values_by_group[(group, task, metric_key)] = min(values)

    best_averages_by_group = {}  # (group, avg_name) -> best value
    for group in (0, 1, 2, 3):
        group_models = [m for m in models if model_group(m) == group]
        for avg_name, avg_tasks, higher_is_better in AVERAGE_COLUMNS:
            avg_values = []
            for model in group_models:
                avg = compute_average(model_results[model], avg_tasks, BENCHMARKS)
                if avg is not None:
                    avg_values.append(avg)
            if avg_values:
                best_averages_by_group[(group, avg_name)] = max(avg_values) if higher_is_better else min(avg_values)

    # Build rows
    for i, model in enumerate(models):
        # Insert midrule when entering a new group (after first row)
        if i > 0 and model_group(model) != model_group(models[i - 1]):
            lines.append(r"\midrule")
        display_name = MODEL_NAME_MAP.get(model, model)
        results = model_results[model].get("results", {})

        group = model_group(model)
        row = [display_name]
        for task, metric_key, _ in BENCHMARKS:
            if task in results and metric_key in results[task]:
                val = results[task][metric_key]
                if isinstance(val, (int, float)):
                    formatted = format_value(val, metric_key)
                    # Bold the best value only for PPL columns, within this group
                    if "perplexity" in metric_key:
                        key = (group, task, metric_key)
                        if key in best_values_by_group and abs(val - best_values_by_group[key]) < 1e-6:
                            formatted = r"\textbf{" + formatted + r"}"
                    row.append(formatted)
                else:
                    row.append("--")
            else:
                row.append("--")

        # Add average columns
        for avg_name, avg_tasks, _ in AVERAGE_COLUMNS:
            avg = compute_average(model_results[model], avg_tasks, BENCHMARKS)
            if avg is not None:
                formatted = f"{avg:.2f}"
                # Bold the best average within this group
                key = (group, avg_name)
                if key in best_averages_by_group and abs(avg - best_averages_by_group[key]) < 1e-6:
                    formatted = r"\textbf{" + formatted + r"}"
                row.append(formatted)
            else:
                row.append("--")

        lines.append(" & ".join(row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def main():
    for size in MODEL_SIZES:
        logs_dir = LOGS_BASE / f"logs-{size}-lm"
        if not logs_dir.is_dir():
            print(f"Skipping {size}: {logs_dir} not found")
            continue
        model_files = find_json_files(logs_dir, size)
        print(f"[{size}] Found {len(model_files)} model results:")
        for model, path in sorted(model_files.items()):
            print(f"  - {model}: {path}")
        if not model_files:
            continue
        model_results = {m: json.load(open(p)) for m, p in model_files.items()}
        latex_table = generate_latex_table(model_results, size)
        output_path = logs_dir / "results_table.tex"
        with open(output_path, "w") as f:
            f.write(latex_table)
        print(f"\nTable saved to: {output_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()
