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
    def __init__(self, path: str | None) -> TorchProfiler:
        self._path = path

        if path is None:
            self._profiler = None
            return

        self.accelerator = Accelerator.get_accelerator()
        self._step = 0

        self._profiler = None
        if self.accelerator == Accelerator.cuda:
            self._profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=5 if ProcessGroupManager.get_global_rank() == 0 else 150000, warmup=5, active=1, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(path),
                record_shapes=True,
                profile_memory=True,
            )

    def __enter__(self):
        if self._path is not None and self.accelerator == Accelerator.tpu:
            self._step = 0
            xla_start_trace(self._path)
        elif self._profiler is not None:
            self._profiler.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._profiler is not None:
            self._profiler.__exit__(exc_type, exc_val, exc_tb)

    def step(self) -> None:
        if self._path is not None and self.accelerator == Accelerator.tpu:
            self._step += 1
            if self._step == 75:
                xla_stop_trace()
        elif self._profiler is not None:
            self._profiler.step()
