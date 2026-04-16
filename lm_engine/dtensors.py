# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import DeviceMesh


def _get_all_markers():
    from .hf_models.parameter import _ALL_MARKERS

    return _ALL_MARKERS


def tensor_to_dtensor(
    tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    current_placement: Placement | list[Placement],
    desired_placement: Placement | list[Placement] | None = None,
    run_check: bool = False,
    copy_marker: bool = True,
) -> DTensor:
    if isinstance(tensor, DTensor):
        return tensor

    if isinstance(current_placement, Placement):
        current_placement = [current_placement]

    dtensor = DTensor.from_local(tensor, device_mesh=device_mesh, run_check=run_check, placements=current_placement)

    if desired_placement is not None:
        if isinstance(desired_placement, Placement):
            desired_placement = [desired_placement]

        dtensor = dtensor.redistribute(device_mesh=device_mesh, placements=desired_placement, async_op=True)

    if copy_marker:
        for marker in _get_all_markers():
            marker_value = getattr(dtensor, marker, None)
            if marker_value is not None:
                setattr(dtensor, marker, marker_value)

    return dtensor


def dtensor_to_tensor(
    dtensor: DTensor,
    device_mesh: DeviceMesh | None = None,
    desired_placement: Placement | list[Placement] | None = None,
    grad_placement: Placement | list[Placement] | None = None,
    copy_marker: bool = True,
) -> torch.Tensor:
    if not isinstance(dtensor, DTensor):
        return dtensor

    if desired_placement is not None:
        if isinstance(desired_placement, Placement):
            desired_placement = [desired_placement]

        assert device_mesh is not None

        dtensor = dtensor.redistribute(device_mesh=device_mesh, placements=desired_placement, async_op=True)

    if grad_placement is not None and isinstance(grad_placement, Placement):
        grad_placement = [grad_placement]

    tensor = dtensor.to_local(grad_placements=grad_placement)

    if copy_marker:
        for marker in _get_all_markers():
            marker_value = getattr(tensor, marker, None)
            if marker_value is not None:
                setattr(tensor, marker, marker_value)

    return tensor


@torch.no_grad()
def modify_state_dict_to_dtensor_dict(module: nn.Module, state_dict: dict, prefix: str, strip_keys: bool) -> dict:
    module_state_dict = module.state_dict()

    result = {}
    for key, tensor in state_dict.items():
        if key.startswith(prefix):
            stripped_key = key.split(prefix)[1] if strip_keys and prefix != "" else key

            param = module_state_dict[stripped_key]
            device_mesh = param.device_mesh
            placements = param.placements

            if isinstance(tensor, DTensor):
                assert tensor.device_mesh == device_mesh
                assert tensor.placements == placements

                result[key] = tensor
            else:
                result[key] = tensor_to_dtensor(tensor, device_mesh=device_mesh, current_placement=placements)

    return result


def use_async_tensor_parallel() -> bool:
    return torch._inductor.config._micro_pipeline_tp
