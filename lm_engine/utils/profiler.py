# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from .accelerator import Accelerator
from .parallel import ProcessGroupManager


class TorchProfiler:
    def __init__(self, path: str | None) -> TorchProfiler:
        if path is None:
            self.torch_profiler = None

        self.accelerator = Accelerator.get_accelerator()

        if self.accelerator == Accelerator.cuda:
            self.torch_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=5 if ProcessGroupManager.get_global_rank() == 0 else 150000, warmup=5, active=1, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(path),
                record_shapes=True,
                profile_memory=True,
            )
        elif self.accelerator == Accelerator.tpu:
            self.torch_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=5 if ProcessGroupManager.get_global_rank() == 0 else 150000, warmup=5, active=1, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(path),
                record_shapes=True,
                profile_memory=True,
            )

    def __enter__(self):
        return None

    def __exit__(self):
        return None
