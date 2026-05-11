# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

# Adapted from torchtitan's graph_trainer JIT compile pipeline:
# https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/graph_trainer
# Original copyright (c) Meta Platforms, Inc. and affiliates.

import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch._functorch.config as functorch_config
from torch._dynamo.backends.common import aot_autograd as aot_autograd_backend
from torch._inductor.config import aten_distributed_optimizations
from torch._inductor.fx_passes.bucketing import is_all_gather_into_tensor, is_wait_tensor
from torch._inductor.fx_passes.micro_pipeline_tp import micro_pipeline_tp_pass
from torch._inductor.fx_passes.overlap_manual_scheduling import manual_overlap_bucketing
from torch._inductor.fx_passes.overlap_scheduling import get_group_name, schedule_overlap_bucketing
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch.utils.checkpoint import CheckpointPolicy


def async_tensor_parallel_pass(gm: torch.fx.GraphModule, example_inputs: tuple) -> torch.fx.GraphModule:
    """Pipeline TP collectives with matmuls via symmetric memory.

    Fuses all-gather + matmul into ``symm_mem.fused_all_gather_matmul`` and
    matmul + reduce-scatter into ``symm_mem.fused_matmul_reduce_scatter``.
    """

    c10d = torch.ops._c10d_functional
    collective_targets = {
        c10d.all_gather_into_tensor.default,
        c10d.reduce_scatter_tensor.default,
    }
    registered: set[str] = set()
    for node in gm.graph.nodes:
        if node.target not in collective_targets:
            continue
        pg = get_group_name(node)
        if pg not in registered:
            registered.add(pg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                enable_symm_mem_for_group(pg)

    micro_pipeline_tp_pass(gm.graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def reassign_to_pg_pass(
    gm: torch.fx.GraphModule,
    example_inputs: Any = None,
    *,
    source_pg_name: str,
    target_pg_name: str,
) -> torch.fx.GraphModule:
    """Rewrite all-gather nodes from ``source_pg_name`` to ``target_pg_name``.

    Must be applied **before** bucketing passes so bucketed AGs inherit the
    new PG.
    """
    count = 0
    for node in gm.graph.nodes:
        if is_all_gather_into_tensor(node) and node.args[2] == source_pg_name:
            node.args = (node.args[0], node.args[1], target_pg_name)
            count += 1
    gm.recompile()
    return gm


def _is_wait_tensor_from_fsdp(node: torch.fx.Node) -> bool:
    """True if ``node`` is the wait_tensor of an FSDP all_gather that can be prefetched.

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

    When ``reshard_after_forward=True``, all_gathers and their immediate
    consumers (wait + post-wait slice + dtype convert) are marked
    MUST_RECOMPUTE so the backward re-runs them rather than holding gathered
    params live through fwd.

    When False, they are marked MUST_SAVE (params stay all-gathered).

    ``ac_graph_id=100000`` prevents the recompute decision from being
    influenced by neighbouring AC regions (partitioner workaround).

    Run as a ``joint_custom_pass`` so AC tags survive the joint->fwd/bwd
    partition.
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

    Calls ``schedule_overlap_bucketing`` with ``collective_bucketing=True``,
    which fuses small all_gathers / reduce_scatters into larger ones and
    reorders around compute. Used as fw/bw compiler in the aot_eager path.
    """
    schedule_overlap_bucketing(gm, collective_bucketing=True)
    gm.recompile()
    return gm


def transformer_block_bucketing_reordering_pass(
    gm: torch.fx.GraphModule,
    example_inputs: Any | None = None,
    *,
    fsdp_manual_buckets: list[list[str] | str],
) -> torch.fx.GraphModule:
    """Manual aten-level bucketing and reordering per transformer block.

    Uses ``manual_overlap_bucketing`` (pytorch/pytorch#165487) which
    groups collectives by module FQN bucket and emits prefetch edges.
    Requires the model to have been annotated with ``annotate_module_fqns``
    before tracing so nodes carry ``module_fqn`` metadata.
    """
    manual_overlap_bucketing(gm, module_bucket_plans=fsdp_manual_buckets, insert_overlap_deps=False)
    gm.recompile()
    return gm


def get_simple_fsdp_compile_backend(
    *,
    fsdp_reshard_after_forward: bool,
    bucketing_mode: str = "none",
    fsdp_manual_buckets: list[list[str] | str] | None = None,
    async_tensor_parallel: bool = False,
    backend: str,
) -> Callable:
    """Build a torch.compile backend that wires SimpleFSDP-aware passes.

    Args:
        fsdp_reshard_after_forward: Install ``annotate_fsdp_all_gather`` as the
            joint_custom_pass so AG nodes get AC-recompute tags before the
            fwd/bwd partition runs.
        bucketing_mode: Collective bucketing strategy.
            - ``"none"``: no bucketing (default).
            - ``"auto"``: global auto-bucketing via ``schedule_overlap_bucketing``.
            - ``"transformer_block"``: per-layer manual bucketing via
              ``manual_overlap_bucketing``. Requires ``fsdp_manual_buckets``
              and that the model was annotated with ``annotate_module_fqns``
              before compilation.
        fsdp_manual_buckets: Module FQN bucket plans for ``"transformer_block"``
            mode. Each entry is a module FQN string or a list of FQN strings
            that should be grouped into one bucket.
        async_tensor_parallel: Apply ``async_tensor_parallel_pass`` to fuse TP
            collectives with matmuls via symmetric memory.
        backend: Underlying torch.compile backend (``"inductor"``,
            ``"aot_eager"``, ``"neuron"``, …).
    """
    if bucketing_mode not in ("none", "auto", "transformer_block"):
        raise ValueError(f"bucketing_mode must be 'none', 'auto', or 'transformer_block', got {bucketing_mode!r}")
    if bucketing_mode == "transformer_block" and not fsdp_manual_buckets:
        raise ValueError("fsdp_manual_buckets must be provided when bucketing_mode='transformer_block'")

    torch._dynamo.config.capture_scalar_outputs = True
    inner_backend = torch._dynamo.lookup_backend(backend)

    if bucketing_mode == "auto":
        if backend == "aot_eager":
            aten_distributed_optimizations.insert_overlap_deps = False

            inner_backend = aot_autograd_backend(
                fw_compiler=autobucketing_reordering_pass,
                bw_compiler=autobucketing_reordering_pass,
                keep_inference_input_mutations=True,
            )
        elif backend in ("inductor", "neuron"):

            def _inductor_autobucketing_pass(gm: torch.fx.Graph) -> torch.fx.GraphModule:
                return schedule_overlap_bucketing(gm.owning_module, collective_bucketing=True)

            aten_distributed_optimizations.collective_bucketing = True
            aten_distributed_optimizations.insert_overlap_deps = True
            torch._inductor.config.allow_buffer_reuse = False
            torch._inductor.config.reorder_for_peak_memory = False
            torch._inductor.config.reorder_for_compute_comm_overlap = False
            torch._inductor.config.post_grad_custom_post_pass = _inductor_autobucketing_pass
        else:
            raise ValueError(f"Unsupported backend {backend!r} for bucketing_mode='auto'")
    elif bucketing_mode == "transformer_block":
        if backend == "aot_eager":
            _tb_pass = partial(transformer_block_bucketing_reordering_pass, fsdp_manual_buckets=fsdp_manual_buckets)

            inner_backend = aot_autograd_backend(
                fw_compiler=_tb_pass, bw_compiler=_tb_pass, keep_inference_input_mutations=True
            )
        elif backend in ("inductor", "neuron"):

            def _inductor_tb_pass(gm: torch.fx.Graph) -> torch.fx.GraphModule:
                return manual_overlap_bucketing(
                    gm.owning_module, module_bucket_plans=fsdp_manual_buckets, insert_overlap_deps=True
                )

            torch._inductor.config.allow_buffer_reuse = False
            torch._inductor.config.reorder_for_peak_memory = False
            torch._inductor.config.reorder_for_compute_comm_overlap = False
            torch._inductor.config.post_grad_custom_post_pass = _inductor_tb_pass
        else:
            raise ValueError(f"Unsupported backend {backend!r} for bucketing_mode='transformer_block'")

    def _joint_ac_pass(gm: torch.fx.GraphModule, example_inputs: Any) -> torch.fx.GraphModule:
        gm = fsdp_reshard_after_fwd_pass(gm, example_inputs, reshard_after_forward=fsdp_reshard_after_forward)
        if async_tensor_parallel:
            gm = async_tensor_parallel_pass(gm, example_inputs)
        return gm

    def _backend_with_passes(*args, **kwargs):
        with functorch_config.patch("joint_custom_pass", _joint_ac_pass):
            return inner_backend(*args, **kwargs)

    return _backend_with_passes
