# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from .accelerator import Accelerator
from .packages import is_torch_xla_available
from .parallel import ProcessGroupManager


if is_torch_xla_available():
    from torch_xla.debug.profiler import start_trace as xla_start_trace
    from torch_xla.debug.profiler import stop_trace as xla_stop_trace


class TorchProfiler:
    def __init__(self, path: str | None, wait: int = 5, active: int = 1, warmup: int = 5) -> TorchProfiler:
        self.path = path

        self.start_step = wait + warmup
        self.end_step = self.start_step + active

        if path is None:
            self._profiler = None
            return

        self.accelerator = Accelerator.get_accelerator()
        self._step = 0

        self._profiler = None
        if self.accelerator != Accelerator.tpu:
            self._profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=wait if ProcessGroupManager.get_global_rank() == 0 else 150000,
                    warmup=warmup,
                    active=active,
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(path),
                record_shapes=True,
                profile_memory=True,
            )

    def __enter__(self):
        if self._profiler is not None:
            self._profiler.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._profiler is not None:
            self._profiler.__exit__(exc_type, exc_val, exc_tb)

    def step(self) -> None:
        if self.path is not None and self.accelerator == Accelerator.tpu:
            self._step += 1
            if self._step == self.start_step:
                xla_start_trace(self.path)
            elif self._step == self.end_step:
                xla_stop_trace()
                self.path = None
        elif self._profiler is not None:
            self._profiler.step()
