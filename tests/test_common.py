# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os
from typing import Callable
from unittest import TestCase, skipUnless

import pytest
import torch


_RUN_SLOW = True if os.getenv("RUN_SLOW", "False").lower() in ["1", "true"] else False


def skip_test_if_device_unavailable(device: torch.device) -> None:
    # convert to str
    if isinstance(device, torch.device):
        device = device.type

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("skipping test because CUDA is unavailable")


class BaseTestCommons(TestCase):
    @staticmethod
    def slow_test(func: Callable) -> Callable:
        return skipUnless(_RUN_SLOW, "skipping slow test since RUN_SLOW=True is not set in the environment")(func)
