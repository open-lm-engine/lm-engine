# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os
from typing import Callable


def get_ptx_from_triton_kernel(kernel: Callable, output_directory: str) -> None:
    os.makedirs(output_directory, exist_ok=True)

    for key, value in kernel.cache[0].items():
        key = key.replace(" ", "_")
        key = key.replace(":", "_")
        value = value.asm["ptx"]

        with open(os.path.join(output_directory, f"{key}.ptx"), "w") as a:
            print(value, file=a)
