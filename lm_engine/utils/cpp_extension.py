# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import os
import types

from torch.utils.cpp_extension import load


def compile_cpp_extension(
    name: str,
    sources: list[str] | str,
    build_directory: str,
    extra_cflags: list[str] = ["-O3", "-Wall", "-shared", "-fPIC", "-fdiagnostics-color"],
    verbose: bool = True,
    distributed: bool = False,
) -> types.ModuleType:
    os.makedirs(build_directory, exist_ok=True)

    def _compile() -> types.ModuleType:
        return load(name, sources=sources, extra_cflags=extra_cflags, build_directory=build_directory, verbose=verbose)

    if distributed:
        from ..accelerator import Communication
        from ..parallel import ProcessGroupManager

        if ProcessGroupManager.get_global_rank() == 0:
            module = _compile()

        Communication.barrier()

        if ProcessGroupManager.get_global_rank() != 0:
            module = _compile()
    else:
        module = _compile()

    return module
