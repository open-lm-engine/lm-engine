# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Callable

import torch
import torch.distributed
from torch.distributed import ProcessGroup
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from .miscellaneous import divide_if_divisible
from .packages import is_torch_xla_available


if is_torch_xla_available():
    from torch_xla.runtime import global_ordinal as xla_global_ordinal
    from torch_xla.runtime import local_ordinal as xla_local_ordinal
    from torch_xla.runtime import world_size as xla_world_size


# general
_MESH: DeviceMesh | None = None
_GLOBAL_RANK: int | None = None
_LOCAL_RANK: int | None = None
_WORLD_SIZE: int | None = None
_CPU_GROUP: ProcessGroup | None = None

# tensor parallel
_TENSOR_PARALLEL_MESH: DeviceMesh | None = None
_TENSOR_PARALLEL_GROUP: ProcessGroup | None = None
_TENSOR_PARALLEL_RANK: int | None = None
_TENSOR_PARALLEL_WORLD_SIZE: int | None = None
_TENSOR_PARALLEL_FIRST_RANK: int | None = None

# pipeline parallel
_PIPELINE_PARALLEL_MESH: DeviceMesh | None = None
_PIPELINE_PARALLEL_GROUP: ProcessGroup | None = None
_PIPELINE_PARALLEL_RANK: int | None = None
_PIPELINE_PARALLEL_WORLD_SIZE: int | None = None

# data parallel
_DATA_PARALLEL_MESH: DeviceMesh | None = None
_DATA_PARALLEL_GROUP: ProcessGroup | None = None
_DATA_PARALLEL_RANK: int | None = None
_DATA_PARALLEL_WORLD_SIZE: int | None = None
_DATA_PARALLEL_REPLICATION_WORLD_SIZE: int | None = None
_DATA_PARALLEL_SHARDING_WORLD_SIZE: int | None = None


class ProcessGroupManager:
    def __init__(
        self,
        tensor_parallel_world_size: int = 1,
        pipeline_parallel_world_size: int = 1,
        data_parallel_replication_world_size: int | None = None,
        data_parallel_sharding_world_size: int | None = None,
        zero_stage: int = 3,
        timeout_minutes: int | None = None,
        use_async_tensor_parallel: bool = False,
    ) -> ProcessGroupManager:
        from .accelerator import Accelerator

        global _MESH
        global _TENSOR_PARALLEL_FIRST_RANK
        global _DATA_PARALLEL_REPLICATION_WORLD_SIZE
        global _DATA_PARALLEL_SHARDING_WORLD_SIZE
        global _CPU_GROUP
        global _GLOBAL_RANK
        global _LOCAL_RANK
        global _WORLD_SIZE
        global _DATA_PARALLEL_WORLD_SIZE

        if timeout_minutes is not None:
            timeout_minutes = timedelta(timeout_minutes)

        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.tpu:
            torch.distributed.init_process_group(backend="xla", init_method="xla://", timeout=timeout_minutes)
            _CPU_GROUP = torch.distributed.new_group(backend="cpu:gloo")

            _GLOBAL_RANK = xla_global_ordinal()
            _LOCAL_RANK = xla_local_ordinal()
            _WORLD_SIZE = xla_world_size()
        else:
            _GLOBAL_RANK = int(os.getenv("RANK", 0))
            _LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
            _WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

            backend = "cpu:gloo"
            if accelerator == Accelerator.cuda:
                backend += ",cuda:nccl"
            elif accelerator == Accelerator.trainium:
                backend += ",neuron:neuron"

            torch.distributed.init_process_group(
                backend=backend, rank=_GLOBAL_RANK, world_size=_WORLD_SIZE, timeout=timeout_minutes
            )

        Accelerator.set_device(_LOCAL_RANK)

        data_parallel_size = _WORLD_SIZE // (tensor_parallel_world_size * pipeline_parallel_world_size)
        assert tensor_parallel_world_size * pipeline_parallel_world_size * data_parallel_size == _WORLD_SIZE

        if zero_stage == 0:
            assert data_parallel_sharding_world_size is None or data_parallel_sharding_world_size == 1

            data_parallel_replication_world_size = data_parallel_size
            data_parallel_sharding_world_size = 1
        else:
            if data_parallel_replication_world_size is None:
                assert data_parallel_sharding_world_size is None

                data_parallel_replication_world_size = 1
                data_parallel_sharding_world_size = data_parallel_size
            else:
                assert data_parallel_sharding_world_size is not None

        assert data_parallel_replication_world_size * data_parallel_sharding_world_size == data_parallel_size

        _DATA_PARALLEL_REPLICATION_WORLD_SIZE = data_parallel_replication_world_size
        _DATA_PARALLEL_SHARDING_WORLD_SIZE = data_parallel_sharding_world_size

        # FIXME unable to use XLA mesh since XLA mesh doesn't support accessing submesh
        _MESH = init_device_mesh(
            "cpu" if accelerator == Accelerator.tpu else Accelerator.get_device_type(),
            (
                pipeline_parallel_world_size,
                data_parallel_replication_world_size,
                data_parallel_sharding_world_size,
                tensor_parallel_world_size,
            ),
            mesh_dim_names=("pp", "ddp", "fsdp", "tp"),
        )

        if use_async_tensor_parallel:
            enable_symm_mem_for_group(ProcessGroupManager.get_tensor_parallel_group().group_name)
            torch._inductor.config._micro_pipeline_tp = True

        if accelerator == Accelerator.tpu:
            assert tensor_parallel_world_size == 1
            assert pipeline_parallel_world_size == 1
        else:
            group = ProcessGroupManager.get_tensor_parallel_group()
            ranks = torch.distributed.get_process_group_ranks(group)

            _TENSOR_PARALLEL_FIRST_RANK = ranks[0]

    @staticmethod
    def is_initialized() -> bool:
        return torch.distributed.is_initialized()

    @staticmethod
    def get_mesh() -> DeviceMesh:
        global _MESH
        return _MESH

    @staticmethod
    def get_global_rank() -> int:
        global _GLOBAL_RANK
        return _GLOBAL_RANK

    @staticmethod
    def get_local_rank() -> int:
        global _LOCAL_RANK
        return _LOCAL_RANK

    @staticmethod
    def get_world_size() -> int:
        global _WORLD_SIZE
        return _WORLD_SIZE

    # tensor parallel
    @staticmethod
    def get_tensor_parallel_mesh() -> DeviceMesh:
        global _TENSOR_PARALLEL_MESH

        if _TENSOR_PARALLEL_MESH is None:
            _TENSOR_PARALLEL_MESH = ProcessGroupManager.get_mesh()["tp"]
        return _TENSOR_PARALLEL_MESH

    @staticmethod
    def get_tensor_parallel_group() -> ProcessGroup:
        global _TENSOR_PARALLEL_GROUP

        if _TENSOR_PARALLEL_GROUP is None:
            _TENSOR_PARALLEL_GROUP = ProcessGroupManager.get_tensor_parallel_mesh().get_group()
        return _TENSOR_PARALLEL_GROUP

    @staticmethod
    def get_tensor_parallel_rank() -> int:
        global _TENSOR_PARALLEL_RANK

        if _TENSOR_PARALLEL_RANK is None:
            _TENSOR_PARALLEL_RANK = ProcessGroupManager.get_tensor_parallel_mesh().get_local_rank()
        return _TENSOR_PARALLEL_RANK

    @contextmanager
    @staticmethod
    def set_dummy_tensor_parallel_rank(rank: int):
        global _TENSOR_PARALLEL_RANK

        original_rank = _TENSOR_PARALLEL_RANK
        _TENSOR_PARALLEL_RANK = rank

        yield

        _TENSOR_PARALLEL_RANK = original_rank

    @staticmethod
    def get_tensor_parallel_world_size() -> int:
        global _TENSOR_PARALLEL_WORLD_SIZE

        if _TENSOR_PARALLEL_WORLD_SIZE is None:
            _TENSOR_PARALLEL_WORLD_SIZE = ProcessGroupManager.get_tensor_parallel_mesh().size()
        return _TENSOR_PARALLEL_WORLD_SIZE

    @contextmanager
    @staticmethod
    def set_dummy_tensor_parallel_world_size(world_size: int):
        global _TENSOR_PARALLEL_WORLD_SIZE

        original_world_size = _TENSOR_PARALLEL_WORLD_SIZE
        _TENSOR_PARALLEL_WORLD_SIZE = world_size

        yield

        _TENSOR_PARALLEL_WORLD_SIZE = original_world_size

    @staticmethod
    def get_tensor_parallel_first_rank() -> int:
        global _TENSOR_PARALLEL_FIRST_RANK
        return _TENSOR_PARALLEL_FIRST_RANK

    @contextmanager
    @staticmethod
    def set_dummy_tensor_parallel_first_rank(rank: int):
        global _TENSOR_PARALLEL_FIRST_RANK

        original_rank = _TENSOR_PARALLEL_FIRST_RANK
        _TENSOR_PARALLEL_FIRST_RANK = rank

        yield

        _TENSOR_PARALLEL_FIRST_RANK = original_rank

    @staticmethod
    def is_tensor_parallel_enabled() -> bool:
        try:
            return ProcessGroupManager.is_initialized() and ProcessGroupManager.get_tensor_parallel_world_size() > 1
        except:
            return False

    @staticmethod
    def is_tensor_parallel_first_rank() -> bool:
        return ProcessGroupManager.get_tensor_parallel_rank() == 0

    # pipeline parallel
    @staticmethod
    def get_pipeline_parallel_mesh() -> DeviceMesh:
        global _PIPELINE_PARALLEL_MESH

        if _PIPELINE_PARALLEL_MESH is None:
            _PIPELINE_PARALLEL_MESH = ProcessGroupManager.get_mesh()["pp"]
        return _PIPELINE_PARALLEL_MESH

    @staticmethod
    def get_pipeline_parallel_group() -> ProcessGroup:
        global _PIPELINE_PARALLEL_GROUP

        if _PIPELINE_PARALLEL_GROUP is None:
            _PIPELINE_PARALLEL_GROUP = ProcessGroupManager.get_pipeline_parallel_mesh().get_group()
        return _PIPELINE_PARALLEL_GROUP

    @staticmethod
    def get_pipeline_parallel_rank() -> int:
        global _PIPELINE_PARALLEL_RANK

        if _PIPELINE_PARALLEL_RANK is None:
            _PIPELINE_PARALLEL_RANK = ProcessGroupManager.get_pipeline_parallel_mesh().get_local_rank()
        return _PIPELINE_PARALLEL_RANK

    @contextmanager
    @staticmethod
    def set_dummy_pipeline_parallel_rank(rank: int):
        global _PIPELINE_PARALLEL_RANK

        original_rank = _PIPELINE_PARALLEL_RANK
        _PIPELINE_PARALLEL_RANK = rank

        yield

        _PIPELINE_PARALLEL_RANK = original_rank

    @staticmethod
    def get_pipeline_parallel_world_size() -> int:
        global _PIPELINE_PARALLEL_WORLD_SIZE

        if _PIPELINE_PARALLEL_WORLD_SIZE is None:
            _PIPELINE_PARALLEL_WORLD_SIZE = ProcessGroupManager.get_pipeline_parallel_mesh().size()
        return _PIPELINE_PARALLEL_WORLD_SIZE

    @contextmanager
    @staticmethod
    def set_dummy_pipeline_parallel_world_size(world_size: int):
        global _PIPELINE_PARALLEL_WORLD_SIZE

        original_world_size = _PIPELINE_PARALLEL_WORLD_SIZE
        _PIPELINE_PARALLEL_WORLD_SIZE = world_size

        yield

        _PIPELINE_PARALLEL_WORLD_SIZE = original_world_size

    # data parallel
    @staticmethod
    def get_data_parallel_mesh() -> DeviceMesh:
        global _DATA_PARALLEL_MESH

        if _DATA_PARALLEL_MESH is None:
            _DATA_PARALLEL_MESH = ProcessGroupManager.get_mesh()["ddp", "fsdp"]
        return _DATA_PARALLEL_MESH

    @staticmethod
    def get_data_parallel_group() -> ProcessGroup:
        global _DATA_PARALLEL_GROUP

        if _DATA_PARALLEL_GROUP is None:
            _DATA_PARALLEL_GROUP = ProcessGroupManager.get_data_parallel_mesh()._flatten().get_group()
        return _DATA_PARALLEL_GROUP

    @staticmethod
    def get_data_parallel_rank() -> int:
        global _DATA_PARALLEL_RANK

        if _DATA_PARALLEL_RANK is None:
            _DATA_PARALLEL_RANK = ProcessGroupManager.get_data_parallel_mesh()._flatten().get_local_rank()
        return _DATA_PARALLEL_RANK

    @contextmanager
    @staticmethod
    def set_dummy_data_parallel_rank(rank: int):
        global _DATA_PARALLEL_RANK

        original_rank = _DATA_PARALLEL_RANK
        _DATA_PARALLEL_RANK = rank

        yield

        _DATA_PARALLEL_RANK = original_rank

    @staticmethod
    def get_data_parallel_world_size() -> int:
        global _DATA_PARALLEL_WORLD_SIZE

        if _DATA_PARALLEL_WORLD_SIZE is None:
            _DATA_PARALLEL_WORLD_SIZE = ProcessGroupManager.get_data_parallel_mesh().size()
        return _DATA_PARALLEL_WORLD_SIZE

    @staticmethod
    def get_data_parallel_replication_world_size() -> int:
        return _DATA_PARALLEL_REPLICATION_WORLD_SIZE

    @staticmethod
    def get_data_parallel_sharding_world_size() -> int:
        return _DATA_PARALLEL_SHARDING_WORLD_SIZE

    @contextmanager
    @staticmethod
    def set_dummy_data_parallel_world_size(world_size: int):
        global _DATA_PARALLEL_WORLD_SIZE

        original_world_size = _DATA_PARALLEL_WORLD_SIZE
        _DATA_PARALLEL_WORLD_SIZE = world_size

        yield

        _DATA_PARALLEL_WORLD_SIZE = original_world_size

    def __str__(self) -> str:
        return str(self.get_mesh())

    @staticmethod
    def destroy_process_groups() -> None:
        if ProcessGroupManager.is_initialized():
            from .communication import Communication

            Communication.barrier()
            torch.distributed.destroy_process_group()

    @staticmethod
    def get_cpu_group() -> ProcessGroup | None:
        global _CPU_GROUP
        return _CPU_GROUP


def run_rank_n(func: Callable, rank: int = 0, barrier: bool = False) -> Callable:
    """wraps a function to run on a single rank, returns a no-op for other ranks

    Args:
        func (Callable): function to wrap
        rank (int, optional): rank on which function should run. Defaults to 0.
        barrier (bool, optional): whether to synchronize the processes at the end of function execution. Defaults to False.

    Returns:
        Callable: wrapped function
    """

    # wrapper function for the rank to execute on
    def func_rank_n(*args, **kwargs):
        global_rank = ProcessGroupManager.get_global_rank()

        if global_rank is None:
            return func(*args, **kwargs)

        output = func(*args, **kwargs) if global_rank == rank else None

        if barrier:
            from .communication import Communication

            Communication.barrier()

        return output

    return func_rank_n


def is_tracking_rank() -> bool:
    return (
        ProcessGroupManager.get_data_parallel_rank() == 0
        and ProcessGroupManager.is_tensor_parallel_first_rank()
        and ProcessGroupManager.get_pipeline_parallel_rank()
        == ProcessGroupManager.get_pipeline_parallel_world_size() - 1
    )


def get_pipeline_stage_ids_on_current_rank(num_pipeline_stages: int) -> int:
    pp_rank = ProcessGroupManager.get_pipeline_parallel_rank()
    pp_world_size = ProcessGroupManager.get_pipeline_parallel_world_size()

    num_pipeline_stages_per_rank = divide_if_divisible(
        num_pipeline_stages,
        pp_world_size,
        "num_pipeline_stages should be divisible by pipeline_parallel_world_size",
    )

    return tuple(pp_rank + i * pp_world_size for i in range(num_pipeline_stages_per_rank))
