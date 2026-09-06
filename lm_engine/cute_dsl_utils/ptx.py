# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os
import re
from pathlib import Path
from typing import Callable

import cutlass


def _sanitize_cache_key(key: object) -> str:
    filename = str(key)
    for ch in " :,()/\\":
        filename = filename.replace(ch, "_")

    return filename


def _keep_tokens() -> set[str]:
    keep = os.environ.get("CUTE_DSL_KEEP", "")
    return {token.strip().lower() for token in keep.split(",") if token.strip()}


CUTE_DSL_DUMP_DIR = "./tmp"


def enable_cute_ptx_dump() -> None:
    # cute-dsl only writes PTX to disk when CUTE_DSL_KEEP=ptx is set, and (with --enable-tvm-ffi,
    # which every op in this repo compiles with) that's the only way to get the PTX at all --
    # there's no __ptx__ attribute on the compiled kernel for the tvm-ffi path.
    #
    # the flag is cached into a live envar object the first time cute-dsl's DSL class is
    # instantiated, so just setting the env var here is not enough once cutlass has already been
    # imported (which it has, by the time an op's compile cache exists) -- the already-created
    # envar also has to be patched in place.
    os.makedirs(CUTE_DSL_DUMP_DIR, exist_ok=True)
    os.environ["CUTE_DSL_DUMP_DIR"] = str(Path(CUTE_DSL_DUMP_DIR).resolve())

    tokens = _keep_tokens()
    tokens.add("ptx")
    os.environ["CUTE_DSL_KEEP"] = ",".join(sorted(tokens))
    os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")

    for cls_name in ("CuTeDSL", "CuteExperimentalDSL"):
        dsl_cls = getattr(cutlass.cutlass_dsl, cls_name, None)
        if dsl_cls is None:
            continue

        try:
            envar = dsl_cls._get_dsl().envar
        except Exception:
            continue

        envar.keep_ptx = True
        if hasattr(envar, "keep_tokens"):
            envar.keep_tokens = frozenset(set(envar.keep_tokens) | {"ptx"})


def _find_ptx_for_function(function_name: str, ptx_paths: list) -> Path | None:
    for ptx_path in ptx_paths:
        if function_name in ptx_path.name:
            return ptx_path

    entry_pattern = re.compile(rf"\.entry\s+{re.escape(function_name)}(?:\s|\()", re.MULTILINE)
    for ptx_path in ptx_paths:
        if entry_pattern.search(ptx_path.read_text(errors="ignore")):
            return ptx_path

    return ptx_paths[0] if len(ptx_paths) == 1 else None


def get_ptx_from_cute_op(op: Callable, output_directory: str) -> None:
    # call enable_cute_ptx_dump(output_directory) and clear op.cache *before* re-running the op,
    # so cute-dsl actually recompiles (and dumps) every config instead of serving already-compiled
    # kernels straight out of the autotune cache.
    cache = getattr(op, "cache", None)
    assert cache is not None, f"{op} has no compile cache, has it been called (and compiled) at least once?"

    dump_dir = Path(os.environ.get("CUTE_DSL_DUMP_DIR", CUTE_DSL_DUMP_DIR))
    ptx_paths = sorted(dump_dir.rglob("*.ptx"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    assert ptx_paths, f"no .ptx files found under {dump_dir}, did you call enable_cute_ptx_dump and re-run {op}?"

    os.makedirs(output_directory, exist_ok=True)

    for key, kernel in cache.items():
        function_name = getattr(kernel, "function_name", None)
        assert function_name is not None, f"compiled kernel for {key} has no function_name to match against"

        ptx_path = _find_ptx_for_function(function_name, ptx_paths)
        assert ptx_path is not None, f"couldn't find a dumped .ptx file for {key} ({function_name}) in {dump_dir}"

        filename = _sanitize_cache_key(key)
        (Path(output_directory) / f"{filename}.ptx").write_text(ptx_path.read_text())
