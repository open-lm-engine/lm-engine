# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from .accelerator import Accelerator
from .packages import is_torch_xla_available
from .parallel import ProcessGroupManager


if is_torch_xla_available():
    import torch_xla.debug.profiler as xp


class TorchProfiler:
    def __init__(self, path: str | None) -> TorchProfiler:
        if path is None:
            self._profiler = None
            return

        self.accelerator = Accelerator.get_accelerator()
        self._path = path

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
        if self._profiler is not None:
            if self.accelerator == Accelerator.tpu:
                xp.start_trace(self._path)
            else:
                self._profiler.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._profiler is not None:
            if self.accelerator == Accelerator.tpu:
                xp.stop_trace()
            else:
                self._profiler.__exit__(exc_type, exc_val, exc_tb)

    def step(self) -> None:
        if self._profiler is not None:
            self._profiler.step()
