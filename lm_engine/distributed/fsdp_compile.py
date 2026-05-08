# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

#
# Adapted from torchtitan's graph_trainer JIT compile pipeline:
# https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/graph_trainer
# Subset: auto-bucketing + reshard-after-forward joint pass for SimpleFSDP.
# Original copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any, Callable

import torch
import torch._functorch.config as functorch_config
from torch._dynamo.backends.common import aot_autograd as aot_autograd_backend
from torch._inductor.config import aten_distributed_optimizations
from torch._inductor.fx_passes.bucketing import is_all_gather_into_tensor, is_wait_tensor
from torch._inductor.fx_passes.overlap_scheduling import (
    schedule_overlap_bucketing,
    schedule_overlap_bucketing_from_inductor_configs,
)
from torch.utils.checkpoint import CheckpointPolicy


def _is_wait_tensor_from_fsdp(node: torch.fx.Node) -> bool:
    """True if `node` is the wait_tensor of an FSDP all_gather that can be prefetched.

    Mirrors torchtitan's heuristic: walk back from the wait through chains of
    single-input ops to a graph placeholder. If reachable, the all_gather is
    prefetchable / recomputable.
    """
    if is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0]):
        n: torch.fx.Node = node.all_input_nodes[0]
        while len(n.all_input_nodes) == 1:
            if n.all_input_nodes[0].op == "placeholder":
                return True
            n = n.all_input_nodes[0]
    return False


def annotate_fsdp_all_gather(gm: torch.fx.GraphModule, reshard_after_forward: bool) -> torch.fx.GraphModule:
    """Tag SimpleFSDP all_gather nodes for reshard-after-forward behavior.

    When `reshard_after_forward=True`, all_gathers and their immediate consumers
    (wait + post-wait slice + dtype convert) are marked MUST_RECOMPUTE, so the
    backward re-runs them rather than holding gathered params live through fwd.

    When False, they are marked MUST_SAVE (params stay all-gathered).

    `ac_graph_id=100000` ensures the recompute decision is not influenced by
    neighboring AC regions (a partitioner workaround).

    Run as a `joint_custom_pass` so AC tags survive the joint -> fwd/bwd partition.
    """
    graph = gm.graph

    def force_recompute_node(node: torch.fx.Node) -> None:
        if reshard_after_forward:
            node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        else:
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        node.meta["ac_graph_id"] = 100000

    # Workaround for https://github.com/pytorch/pytorch/issues/136433
    for node in graph.nodes:
        if _is_wait_tensor_from_fsdp(node):
            ag_node = node.args[0]
            force_recompute_node(ag_node)
            force_recompute_node(node)
            for user in node.users:
                if user.op == "call_function" and user.target == torch.ops.aten.slice.Tensor:
                    force_recompute_node(user)
            if (
                ag_node.all_input_nodes[0].op == "call_function"
                and ag_node.args[0].target == torch.ops.prims.convert_element_type.default
            ):
                force_recompute_node(ag_node.all_input_nodes[0])

    return gm


def fsdp_reshard_after_fwd_pass(
    gm: torch.fx.GraphModule, example_inputs: Any | None = None, *, reshard_after_forward: bool
) -> torch.fx.GraphModule:
    gm = annotate_fsdp_all_gather(gm, reshard_after_forward)
    gm.recompile()
    return gm


def autobucketing_reordering_pass(gm: torch.fx.GraphModule, example_inputs: Any | None = None) -> torch.fx.GraphModule:
    """Bucket and reorder collectives for compute/comm overlap.

    Calls `schedule_overlap_bucketing` with `collective_bucketing=True`, which
    fuses small all_gathers / reduce_scatters into larger ones and reorders
    around compute. Used as fw/bw compiler in the aot_eager path.
    """

    schedule_overlap_bucketing(gm, collective_bucketing=True)
    gm.recompile()
    return gm


def get_simple_fsdp_compile_backend(
    *,
    fsdp_reshard_after_forward: bool,
    auto_bucketing: bool,
    backend: str = "inductor",
) -> Callable:
    """Build a torch.compile backend that wires SimpleFSDP-aware passes.

    - `fsdp_reshard_after_forward`: install `annotate_fsdp_all_gather` as the
      joint_custom_pass so AG nodes get AC-recompute tags before the fwd/bwd
      partition runs.
    - `auto_bucketing`: enable inductor's distributed-collective bucketing
      scheduler. Conflicts with `reorder_for_compute_comm_overlap`, so flips
      that off; uses `post_grad_custom_post_pass` (inductor) or `aot_autograd`
      with the bucketing pass as fw/bw compiler (aot_eager).
    """
    inner_backend = torch._dynamo.lookup_backend(backend)

    if auto_bucketing:
        if backend == "aot_eager":
            aten_distributed_optimizations.insert_overlap_deps = False

            inner_backend = aot_autograd_backend(
                fw_compiler=autobucketing_reordering_pass,
                bw_compiler=autobucketing_reordering_pass,
                keep_inference_input_mutations=True,
            )
        elif backend == "inductor":

            def _inductor_autobucketing_pass(gm: torch.fx.Graph) -> torch.fx.GraphModule:
                return schedule_overlap_bucketing_from_inductor_configs(gm.owning_module)

            aten_distributed_optimizations.collective_bucketing = True
            aten_distributed_optimizations.insert_overlap_deps = True

            torch._inductor.config.allow_buffer_reuse = False
            torch._inductor.config.reorder_for_peak_memory = False
            torch._inductor.config.reorder_for_compute_comm_overlap = False
            torch._inductor.config.post_grad_custom_post_pass = _inductor_autobucketing_pass
        else:
            raise ValueError(f"Unsupported backend {backend} for auto_bucketing")

    def _joint_ac_pass(gm: torch.fx.GraphModule, example_inputs: Any) -> torch.fx.GraphModule:
        return fsdp_reshard_after_fwd_pass(gm, example_inputs, reshard_after_forward=fsdp_reshard_after_forward)

    def _backend_with_passes(*args, **kwargs):
        with functorch_config.patch("joint_custom_pass", _joint_ac_pass):
            return inner_backend(*args, **kwargs)

    return _backend_with_passes
