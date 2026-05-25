# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

import torch
import torch.distributed
from torch.distributed import ProcessGroup
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from ..accelerator import Accelerator
from ..enums import ContextParallelLoadBalancerMethod
from ..utils import is_torch_xla_available


if is_torch_xla_available():
    from torch_xla.runtime import global_ordinal as xla_global_ordinal
    from torch_xla.runtime import local_ordinal as xla_local_ordinal
    from torch_xla.runtime import world_size as xla_world_size


@dataclass
class _Mesh:
    mesh: DeviceMesh | None = None
    group: ProcessGroup | None = None
    rank: int | None = None
    local_rank: int | None = None
    world_size: int | None = None

    def get_mesh(self) -> DeviceMesh:
        return self.mesh

    def get_group(self) -> ProcessGroup:
        if self.group is None:
            self.group = self.mesh.get_group()

        return self.group

    def get_rank(self) -> int:
        return self.rank

    def get_local_rank(self) -> int:
        if self.local_rank is None:
            self.local_rank = self.mesh.get_local_rank()

        return self.local_rank

    def get_world_size(self) -> int:
        if self.world_size is None:
            self.world_size = self.mesh.size()

        return self.world_size

    @contextmanager
    def set_dummy_rank(self, rank: int):
        original_rank = self.rank
        self.rank = rank

        yield

        self.rank = original_rank

    @contextmanager
    def set_dummy_local_rank(self, local_rank: int):
        original_local_rank = self.local_rank
        self.local_rank = local_rank

        yield

        self.local_rank = original_local_rank

    @contextmanager
    def set_dummy_world_size(self, world_size: int):
        original_world_size = self.world_size
        self.world_size = world_size

        yield

        self.world_size = original_world_size


_DENSE_MESH = _Mesh()
_TENSOR_PARALLEL_MESH = _Mesh()
_TENSOR_PARALLEL_FIRST_RANK: int | None = None
_PIPELINE_PARALLEL_MESH = _Mesh()
_DATA_PARALLEL_MESH = _Mesh()
_DATA_PARALLEL_REPLICATION_WORLD_SIZE: int | None = None
_DATA_PARALLEL_SHARDING_WORLD_SIZE: int | None = None
_CPU_GROUP: ProcessGroup | None = None
_DATA_LOADING_MESH = _Mesh()
_CONTEXT_PARALLEL_MESH = _Mesh()
_CONTEXT_PARALLEL_LOAD_BALANCING_METHOD: ContextParallelLoadBalancerMethod | None = None


class ProcessGroupManager:
    def __init__(
        self,
        tensor_parallel_world_size: int = 1,
        pipeline_parallel_world_size: int = 1,
        data_parallel_replication_world_size: int | None = None,
        data_parallel_sharding_world_size: int | None = None,
        context_parallel_world_size: int = 1,
        context_parallel_load_balancing_method: ContextParallelLoadBalancerMethod = ContextParallelLoadBalancerMethod.headtail,
        zero_stage: int = 3,
        timeout_minutes: int | None = None,
        use_async_tensor_parallel: bool = False,
    ) -> ProcessGroupManager:
        global _DENSE_MESH
        global _TENSOR_PARALLEL_MESH
        global _TENSOR_PARALLEL_FIRST_RANK
        global _PIPELINE_PARALLEL_MESH
        global _DATA_PARALLEL_MESH
        global _DATA_PARALLEL_REPLICATION_WORLD_SIZE
        global _DATA_PARALLEL_SHARDING_WORLD_SIZE
        global _CPU_GROUP
        global _DATA_LOADING_MESH
        global _CONTEXT_PARALLEL_MESH
        global _CONTEXT_PARALLEL_LOAD_BALANCING_METHOD

        _CONTEXT_PARALLEL_LOAD_BALANCING_METHOD = context_parallel_load_balancing_method

        if timeout_minutes is not None:
            timeout_minutes = timedelta(timeout_minutes)

        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.tpu:
            torch.distributed.init_process_group(backend="xla", init_method="xla://", timeout=timeout_minutes)
            _CPU_GROUP = torch.distributed.new_group(backend="cpu:gloo")

            global_rank = xla_global_ordinal()
            local_rank = xla_local_ordinal()
            world_size = xla_world_size()
        else:
            global_rank = int(os.getenv("RANK", 0))
            local_rank = int(os.getenv("LOCAL_RANK", 0))
            world_size = int(os.getenv("WORLD_SIZE", 1))

            backend = "cpu:gloo"
            if accelerator == Accelerator.cuda:
                backend += ",cuda:nccl"
            elif accelerator == Accelerator.trainium:
                backend += ",neuron:neuron"

            torch.distributed.init_process_group(
                backend=backend, rank=global_rank, world_size=world_size, timeout=timeout_minutes
            )

        Accelerator.set_device(local_rank)

        data_loading_world_size = world_size // (
            tensor_parallel_world_size * pipeline_parallel_world_size * context_parallel_world_size
        )

        assert (
            tensor_parallel_world_size
            * pipeline_parallel_world_size
            * data_loading_world_size
            * context_parallel_world_size
            == world_size
        )

        if zero_stage == 0:
            assert data_parallel_sharding_world_size is None or data_parallel_sharding_world_size == 1

            data_parallel_replication_world_size = data_loading_world_size * context_parallel_world_size
            data_parallel_sharding_world_size = 1
        else:
            if data_parallel_replication_world_size is None:
                assert data_parallel_sharding_world_size is None

                data_parallel_replication_world_size = 1
                data_parallel_sharding_world_size = data_loading_world_size * context_parallel_world_size
            else:
                assert data_parallel_sharding_world_size is not None

        assert (
            data_parallel_replication_world_size * data_parallel_sharding_world_size
            == data_loading_world_size * context_parallel_world_size
        )

        assert data_parallel_replication_world_size is not None
        assert data_parallel_sharding_world_size is not None

        _DATA_PARALLEL_REPLICATION_WORLD_SIZE = data_parallel_replication_world_size
        _DATA_PARALLEL_SHARDING_WORLD_SIZE = data_parallel_sharding_world_size

        device_type = "cpu" if accelerator in [Accelerator.mps, Accelerator.tpu] else Accelerator.get_device_type()

        # FIXME unable to use XLA mesh since XLA mesh doesn't support accessing submesh
        _DENSE_MESH = _Mesh(
            mesh=init_device_mesh(
                "cpu" if accelerator in [Accelerator.mps, Accelerator.tpu] else Accelerator.get_device_type(),
                (
                    pipeline_parallel_world_size,
                    data_parallel_replication_world_size,
                    data_parallel_sharding_world_size,
                    tensor_parallel_world_size,
                ),
                mesh_dim_names=("pp", "ddp", "fsdp", "tp"),
            ),
            rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
        )

        _TENSOR_PARALLEL_MESH = _Mesh(mesh=_DENSE_MESH.get_mesh()["tp"])
        _PIPELINE_PARALLEL_MESH = _Mesh(mesh=_DENSE_MESH.get_mesh()["pp"])

        dp_submesh = _DENSE_MESH.get_mesh()["ddp", "fsdp"]
        _DATA_PARALLEL_MESH = _Mesh(
            mesh=dp_submesh, group=dp_submesh._flatten().get_group(), local_rank=dp_submesh._flatten().get_local_rank()
        )

        # separate mesh that exposes the cp dimension explicitly, used to form the CP process group
        _DATA_LOADING_MESH = _Mesh(
            mesh=init_device_mesh(
                device_type,
                (
                    pipeline_parallel_world_size,
                    data_loading_world_size,
                    context_parallel_world_size,
                    tensor_parallel_world_size,
                ),
                mesh_dim_names=("pp", "batch", "cp", "tp"),
            )
        )

        _CONTEXT_PARALLEL_MESH = _Mesh(mesh=_DATA_LOADING_MESH.get_mesh()["cp"])

        if use_async_tensor_parallel:
            enable_symm_mem_for_group(ProcessGroupManager.get_tensor_parallel_group().group_name)
            torch._inductor.config._micro_pipeline_tp = True

        if accelerator == Accelerator.tpu:
            assert tensor_parallel_world_size == 1
            assert pipeline_parallel_world_size == 1
            assert context_parallel_world_size == 1
        else:
            group = ProcessGroupManager.get_tensor_parallel_group()
            ranks = torch.distributed.get_process_group_ranks(group)

            _TENSOR_PARALLEL_FIRST_RANK = ranks[0]

    @staticmethod
    def is_initialized() -> bool:
        return torch.distributed.is_initialized()

    @staticmethod
    def get_dense_mesh() -> DeviceMesh:
        return _DENSE_MESH.get_mesh()

    @staticmethod
    def get_global_rank() -> int:
        return _DENSE_MESH.get_rank()

    @staticmethod
    def get_local_rank() -> int:
        return _DENSE_MESH.get_local_rank()

    @staticmethod
    def get_world_size() -> int:
        return _DENSE_MESH.get_world_size()

    # tensor parallel
    @staticmethod
    def get_tensor_parallel_mesh() -> DeviceMesh:
        return _TENSOR_PARALLEL_MESH.get_mesh()

    @staticmethod
    def get_tensor_parallel_group() -> ProcessGroup:
        return _TENSOR_PARALLEL_MESH.get_group()

    @staticmethod
    def get_tensor_parallel_rank() -> int:
        return _TENSOR_PARALLEL_MESH.get_local_rank()

    @contextmanager
    @staticmethod
    def set_dummy_tensor_parallel_rank(rank: int):
        with _TENSOR_PARALLEL_MESH.set_dummy_local_rank(rank):
            yield

    @staticmethod
    def get_tensor_parallel_world_size() -> int:
        return _TENSOR_PARALLEL_MESH.get_world_size()

    @contextmanager
    @staticmethod
    def set_dummy_tensor_parallel_world_size(world_size: int):
        with _TENSOR_PARALLEL_MESH.set_dummy_world_size(world_size):
            yield

    @staticmethod
    def get_tensor_parallel_first_rank() -> int:
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
        return _PIPELINE_PARALLEL_MESH.get_mesh()

    @staticmethod
    def get_pipeline_parallel_group() -> ProcessGroup:
        return _PIPELINE_PARALLEL_MESH.get_group()

    @staticmethod
    def get_pipeline_parallel_rank() -> int:
        return _PIPELINE_PARALLEL_MESH.get_local_rank()

    @contextmanager
    @staticmethod
    def set_dummy_pipeline_parallel_rank(rank: int):
        with _PIPELINE_PARALLEL_MESH.set_dummy_local_rank(rank):
            yield

    @staticmethod
    def get_pipeline_parallel_world_size() -> int:
        return _PIPELINE_PARALLEL_MESH.get_world_size()

    @contextmanager
    @staticmethod
    def set_dummy_pipeline_parallel_world_size(world_size: int):
        with _PIPELINE_PARALLEL_MESH.set_dummy_world_size(world_size):
            yield

    @staticmethod
    def get_data_loading_rank() -> int:
        return _DATA_LOADING_MESH.get_mesh()["batch"].get_local_rank()

    @staticmethod
    def get_data_loading_world_size() -> int:
        return _DATA_LOADING_MESH.get_mesh()["batch"].size()

    # data parallel
    @staticmethod
    def get_data_parallel_mesh() -> DeviceMesh:
        return _DATA_PARALLEL_MESH.get_mesh()

    @staticmethod
    def get_data_parallel_group() -> ProcessGroup:
        return _DATA_PARALLEL_MESH.get_group()

    @staticmethod
    def get_data_parallel_rank() -> int:
        return _DATA_PARALLEL_MESH.get_local_rank()

    @contextmanager
    @staticmethod
    def set_dummy_data_parallel_rank(rank: int):
        with _DATA_PARALLEL_MESH.set_dummy_local_rank(rank):
            yield

    @staticmethod
    def get_data_parallel_world_size() -> int:
        return _DATA_PARALLEL_MESH.get_world_size()

    @staticmethod
    def get_data_parallel_replication_world_size() -> int:
        return _DATA_PARALLEL_REPLICATION_WORLD_SIZE

    @staticmethod
    def get_data_parallel_sharding_world_size() -> int:
        return _DATA_PARALLEL_SHARDING_WORLD_SIZE

    @contextmanager
    @staticmethod
    def set_dummy_data_parallel_world_size(world_size: int):
        with _DATA_PARALLEL_MESH.set_dummy_world_size(world_size):
            yield

    # context parallel
    @staticmethod
    def get_context_parallel_mesh() -> DeviceMesh:
        return _CONTEXT_PARALLEL_MESH.get_mesh()

    @staticmethod
    def get_context_parallel_group() -> ProcessGroup:
        return _CONTEXT_PARALLEL_MESH.get_group()

    @staticmethod
    def get_context_parallel_rank() -> int:
        return _CONTEXT_PARALLEL_MESH.get_local_rank()

    @contextmanager
    @staticmethod
    def set_dummy_context_parallel_rank(rank: int):
        with _CONTEXT_PARALLEL_MESH.set_dummy_local_rank(rank):
            yield

    @staticmethod
    def get_context_parallel_world_size() -> int:
        return _CONTEXT_PARALLEL_MESH.get_world_size()

    @contextmanager
    @staticmethod
    def set_dummy_context_parallel_world_size(world_size: int):
        with _CONTEXT_PARALLEL_MESH.set_dummy_world_size(world_size):
            yield

    @staticmethod
    def is_context_parallel_enabled() -> bool:
        try:
            return ProcessGroupManager.is_initialized() and ProcessGroupManager.get_context_parallel_world_size() > 1
        except:
            return False

    @staticmethod
    def is_context_parallel_first_rank() -> bool:
        return ProcessGroupManager.get_context_parallel_rank() == 0

    @staticmethod
    def get_context_parallel_load_balancing_method() -> ContextParallelLoadBalancerMethod | None:
        return _CONTEXT_PARALLEL_LOAD_BALANCING_METHOD

    def __str__(self) -> str:
        return str({"dense_mesh": (self.get_dense_mesh()), "dataloading_mesh": _DATA_LOADING_MESH})

    @staticmethod
    def destroy_process_groups() -> None:
        if ProcessGroupManager.is_initialized():
            ProcessGroupManager.barrier()
            torch.distributed.destroy_process_group()

    @staticmethod
    def get_cpu_group() -> ProcessGroup | None:
        return _CPU_GROUP

    @staticmethod
    def broadcast_object(obj: Any, src: int, group: ProcessGroup) -> Any:
        if ProcessGroupManager.get_global_rank() != src:
            obj = None

        object_list = [obj]
        torch.distributed.broadcast_object_list(object_list, src=src, group=group)
        obj = object_list[0]

        return obj

    @staticmethod
    def barrier() -> None:
        torch.distributed.barrier()

        if Accelerator.get_accelerator() == Accelerator.tpu:
            torch.distributed.barrier(ProcessGroupManager.get_cpu_group())


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
            ProcessGroupManager.barrier()

        return output

    return func_rank_n
