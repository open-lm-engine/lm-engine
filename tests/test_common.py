# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os
from itertools import product
from typing import Any, Callable
from unittest import TestCase, skipUnless

import torch


_RUN_SLOW = True if os.getenv("RUN_SLOW", "False").lower() in ["1", "true"] else False


class BaseTestCommons(TestCase):
    @staticmethod
    def get_all_devices() -> list[torch.device]:
        return [torch.device("cpu"), torch.device("cuda")]

    @staticmethod
    def get_dtypes() -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    def make_args_matrix(*args_lists) -> list[Any]:
        return [p for p in product(*args_lists)]

    def skip_test_if_device_unavailable(self, device: torch.device) -> None:
        # convert to str
        if isinstance(device, torch.device):
            device = device.type

        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("skipping test because CUDA is unavailable")

    @staticmethod
    def slow_test(func: Callable) -> Callable:
        return skipUnless(_RUN_SLOW, "skipping slow test since RUN_SLOW=True is not set in the environment")(func)
