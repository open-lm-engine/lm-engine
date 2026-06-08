# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Einops-pattern helpers for MuonHSplit.

These drive the per-head "split" path: each splittable param carries a `_split_spec`
attribute `(tag, shape, axes)`, and the optimizer's `patterns` config maps each tag to an
einops "lhs -> rhs" string. The functions here parse/validate those patterns at startup and
resolve the active (pattern, axes) for a param at step time. All pure string/dict logic —
no tensor math beyond a meta-tensor dry-run in verify_and_log_specs.
"""

from __future__ import annotations

import logging

import torch


def parse_lhs(lhs: str) -> list[tuple[str, ...]]:
    # Tokenise the LHS of an einops pattern into per-dim letter tuples.
    # Example: "(K B H) D" -> [("K","B","H"), ("D",)]
    tokens = lhs.replace("(", " ( ").replace(")", " ) ").split()
    dims: list[tuple[str, ...]] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "(":
            j = tokens.index(")", i)
            dims.append(tuple(tokens[i + 1 : j]))
            i = j + 1
        elif t == ")":
            raise ValueError(f"unexpected ')' in pattern LHS: {lhs!r}")
        else:
            dims.append((t,))
            i += 1
    return dims


def check_lhs_matches_shape(pattern: str, axes: dict, shape: tuple[int, ...]) -> None:
    # Validate at startup that (axes, shape) is consistent with the pattern LHS:
    # dim count matches, no orphan axes, and each LHS dim's known-axes product
    # equals (or cleanly divides, with one unknown to infer) the actual size.
    lhs = pattern.split("->")[0].strip()
    dims = parse_lhs(lhs)
    if len(dims) != len(shape):
        raise ValueError(f"pattern LHS {lhs!r} has {len(dims)} dims, but spec shape {shape} has {len(shape)}")
    lhs_letters = {a for d in dims for a in d}
    orphans = sorted(set(axes) - lhs_letters)
    if orphans:
        raise ValueError(
            f"axes {orphans} supplied to set_split_spec but not referenced on LHS of pattern {pattern!r}; "
            f"einops would reject these as unused kwargs"
        )
    for i, (letters, actual) in enumerate(zip(dims, shape)):
        known = {a: axes[a] for a in letters if a in axes}
        unknown = [a for a in letters if a not in axes]
        prod_known = 1
        for v in known.values():
            prod_known *= v
        if not unknown:
            if prod_known != actual:
                raise ValueError(
                    f"LHS dim {i} {letters}: product of axes {known} = {prod_known}, "
                    f"but param shape[{i}] = {actual}"
                )
        elif len(unknown) == 1:
            if actual % prod_known != 0:
                raise ValueError(
                    f"LHS dim {i} {letters}: known axes {known} product = {prod_known} "
                    f"does not divide param shape[{i}] = {actual}; cannot infer {unknown[0]}"
                )
        else:
            raise ValueError(
                f"LHS dim {i} {letters}: more than one unknown axis {unknown}; "
                f"set_split_spec must supply at least all but one of them"
            )


def reverse_pattern(pattern: str) -> str:
    # Flip an einops pattern around the arrow: "a -> b" becomes "b -> a".
    lhs, rhs = pattern.split("->")
    return f"{rhs.strip()} -> {lhs.strip()}"


def active_spec(parameter, targets: set[str], patterns: dict[str, str]) -> tuple | None:
    # Resolve the (pattern, axes) a param should split with, or None if it isn't an active
    # split target. Returns None when the param has no spec, its tag isn't targeted, or no
    # pattern is configured for its tag.
    attr = getattr(parameter, "_split_spec", None)
    if attr is None:
        return None
    tag, _shape, axes = attr
    if tag not in targets:
        return None
    pattern = patterns.get(tag)
    if pattern is None:
        return None
    return pattern, dict(axes)


def _route_label(parameter, is_hyperball: bool, targets: set[str], patterns: dict[str, str]) -> str:
    # Human-readable label for which optimizer path a param takes.
    if not is_hyperball:
        return "adamw"
    if getattr(parameter, "_per_row_hyperball", False):
        return "per-row"
    spec = getattr(parameter, "_split_spec", None)
    if spec is not None:
        tag = spec[0]
        if tag in targets and patterns.get(tag) is not None:
            return f"spec:{tag}"
        return f"NS (spec '{tag}' inactive)"
    return "NS"


def _mask_layer_index(name: str) -> str:
    # Collapse digit runs to '#' so per-layer params group into one table row.
    # "...h.0.mlp.b_proj.weight" -> "...h.#.mlp.b_proj.weight"
    import re

    return re.sub(r"\d+", "#", name)


def _split_shape(pattern: str, axes: dict, shape: tuple[int, ...]) -> tuple[int, ...]:
    # Per-param shape after applying the einops split (meta dry-run, no real compute).
    from einops import rearrange

    return tuple(rearrange(torch.empty(shape, device="meta"), pattern, **axes).shape)


def _render_table(title: str, headers: list[str], rows: list[list[str]], aligns: list[str]) -> str:
    # Render a clean box-drawn table. aligns[i] is "l" or "r" per column.
    cols = len(headers)
    widths = [len(headers[c]) for c in range(cols)]
    for row in rows:
        for c in range(cols):
            widths[c] = max(widths[c], len(row[c]))

    def fmt(cells: list[str]) -> str:
        parts = []
        for c in range(cols):
            parts.append(cells[c].rjust(widths[c]) if aligns[c] == "r" else cells[c].ljust(widths[c]))
        return "│ " + " │ ".join(parts) + " │"

    def rule(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (widths[c] + 2) for c in range(cols)) + right

    out = [
        "",
        f"  {title}",
        rule("┌", "┬", "┐"),
        fmt(headers),
        rule("├", "┼", "┤"),
    ]
    out += [fmt(r) for r in rows]
    out.append(rule("└", "┴", "┘"))
    return "\n".join(out)


def log_param_routing_table(named_params, targets: set[str], patterns: dict[str, str]) -> None:
    """Log a formatted table of how each param is routed by the optimizer.

    named_params: iterable of (name, param, is_hyperball). Rows are collapsed across layers
    (digit runs masked to '#') with a count, so a 24-layer model prints ~one row per distinct
    routing instead of hundreds. Spec rows also show the einops pattern and the per-param
    split shape (computed via a meta-tensor dry-run).
    """
    from ..logging_utils import log_rank_0

    # Collapse identical rows; key carries everything shown so distinct routings stay separate.
    counts: dict[tuple, int] = {}
    order: list[tuple] = []
    for name, p, is_hyperball in named_params:
        route = _route_label(p, is_hyperball, targets, patterns)
        einops_str, split_str = "", ""
        if route.startswith("spec:"):
            tag, shape, axes = p._split_spec
            pattern = patterns[tag]
            einops_str = pattern
            split_str = str(_split_shape(pattern, axes, tuple(shape)))
        key = (_mask_layer_index(name), route, str(tuple(p.shape)), einops_str, split_str)
        if key not in counts:
            order.append(key)
        counts[key] = counts.get(key, 0) + 1

    headers = ["param (layer idx → #)", "count", "route", "shape", "einops", "split shape"]
    aligns = ["l", "r", "l", "l", "l", "l"]
    rows = [[k[0], str(counts[k]), k[1], k[2], k[3], k[4]] for k in order]
    log_rank_0(logging.INFO, _render_table("[MuonHSplit] parameter routing", headers, rows, aligns))


def log_ns_batches(batches: list[tuple[str, tuple[int, ...], list[str]]]) -> None:
    """Log the shape of each tensor fed into NS and the param names stacked into it.

    batches: list of (group_label, stacked_input_shape, [param_names]). Param names are
    collapsed across layers (digit runs masked) with a count.
    """
    from ..logging_utils import log_rank_0

    rows = []
    for label, shape, names in batches:
        collapsed: dict[str, int] = {}
        order: list[str] = []
        for n in names:
            m = _mask_layer_index(n)
            if m not in collapsed:
                order.append(m)
            collapsed[m] = collapsed.get(m, 0) + 1
        members = ", ".join(f"{m} (×{collapsed[m]})" if collapsed[m] > 1 else m for m in order)
        rows.append([label, str(tuple(shape)), str(len(names)), members])

    headers = ["NS group", "NS input shape", "#params", "members (layer idx → #)"]
    aligns = ["l", "l", "r", "l"]
    log_rank_0(logging.INFO, _render_table("[MuonHSplit] Newton-Schulz batched inputs", headers, rows, aligns))


def verify_and_log_specs(param_groups, targets: set[str], patterns: dict[str, str], mode: str) -> None:
    # Startup validation + logging of the split configuration. Walks every hyperball-group
    # param, checks each active spec's pattern against its shape (and does an einops dry-run on
    # a meta tensor), logs a per-tag summary, and raises if any configured target has no param.
    from einops import rearrange

    from ..logging_utils import log_rank_0

    total = 0
    with_spec_active = 0
    with_spec_inactive: dict[str, int] = {}
    no_spec_by_shape: dict[tuple, int] = {}
    seen: set[tuple[str, tuple[int, ...]]] = set()
    found_for_target = {t: False for t in targets}

    for group in param_groups:
        if not group.get("hyperball", False):
            continue
        for p in group["params"]:
            total += 1
            attr = getattr(p, "_split_spec", None)
            if attr is None:
                no_spec_by_shape[tuple(p.shape)] = no_spec_by_shape.get(tuple(p.shape), 0) + 1
                continue
            tag, shape, axes = attr
            if tag not in targets:
                with_spec_inactive[tag] = with_spec_inactive.get(tag, 0) + 1
                continue
            with_spec_active += 1
            found_for_target[tag] = True
            pattern = patterns.get(tag)
            if pattern is None:
                raise ValueError(f"MuonHSplit: no pattern in config for active tag {tag!r}")
            if tuple(p.shape) != tuple(shape):
                raise ValueError(
                    f"MuonHSplit: param shape {tuple(p.shape)} does not match spec shape "
                    f"{tuple(shape)} for tag {tag!r}"
                )
            check_lhs_matches_shape(pattern, axes, tuple(shape))
            key = (tag, tuple(shape))
            if key in seen:
                continue
            seen.add(key)
            dummy = torch.empty(shape, device="meta")
            try:
                out = rearrange(dummy, pattern, **axes)
            except Exception as e:
                raise ValueError(
                    f"MuonHSplit: einops failed for tag {tag!r} shape {tuple(shape)} "
                    f"pattern {pattern!r} axes {axes}: {e}"
                ) from e
            log_rank_0(
                logging.INFO,
                f"[MuonHSplit] active tag={tag!r} weight_shape={tuple(shape)} "
                f"einops={pattern!r} axes={axes} -> batched={tuple(out.shape)}",
            )

    log_rank_0(
        logging.INFO,
        f"[MuonHSplit] hyperball group: total={total} active_split={with_spec_active} "
        f"inactive_tags={with_spec_inactive} no_spec_shapes={dict(sorted(no_spec_by_shape.items()))} "
        f"targets={sorted(targets)} mode={mode!r}",
    )

    missing = [t for t, found in found_for_target.items() if not found]
    if missing:
        raise ValueError(
            f"MuonHSplit: target tag(s) {missing} configured but no param has them. "
            f"Check that the model attaches the marker via set_split_spec()."
        )
