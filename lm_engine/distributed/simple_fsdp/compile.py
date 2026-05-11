# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

# Adapted from torchtitan's graph_trainer JIT compile pipeline:
# https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/graph_trainer
# Original copyright (c) Meta Platforms, Inc. and affiliates.

import heapq
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import torch
import torch._functorch.config as functorch_config
from torch._dynamo.backends.common import aot_autograd as aot_autograd_backend
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch._inductor.config import aten_distributed_optimizations
from torch._inductor.fx_passes.bucketing import BucketMode, is_all_gather_into_tensor, is_wait_tensor
from torch._inductor.fx_passes.overlap_manual_scheduling import ManualOverlapScheduler, manual_overlap_bucketing
from torch._inductor.fx_passes.overlap_scheduling import is_compute_node, schedule_overlap_bucketing
from torch.distributed.device_mesh import DeviceMesh
from torch.fx.traceback import annotate_fn
from torch.utils._ordered_set import OrderedSet
from torch.utils.checkpoint import CheckpointPolicy


# Metadata key used to store the module FQN on FX nodes.
_MODULE_FQN = "module_fqn"
_NOT_IN_LAYERS = -1

# Maps original FSDP group_name -> extra PG group_name
_EXTRA_FSDP_PG_REGISTRY: dict[str, str] = {}


def _is_backward_node(node: torch.fx.Node) -> bool:
    return node.meta.get("autograd_backward", False)


def _is_recomputed_node(node: torch.fx.Node) -> bool:
    # Recomputed nodes (from SAC) should carry autograd_backward=True but
    # remat_using_tags_for_fwd_loss_bwd_graph copies metadata from the original
    # forward node. Tag recomputed nodes by name suffix as a workaround.
    return node.name.endswith("_recomputed")


def _get_layer_id(node: torch.fx.Node) -> int:
    """Return the transformer layer index from the node's module_fqn metadata.

    Nodes under ``layers.<N>`` return N; all others return _NOT_IN_LAYERS.
    """
    fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
    parts = fqn.split(".")
    if parts[0] == "layers" and len(parts) >= 2:
        with suppress(ValueError):
            return int(parts[1])
    return _NOT_IN_LAYERS


def annotate_module_fqns(model: torch.nn.Module) -> None:
    """Wrap every named submodule's forward with its FQN so FX nodes carry
    ``node.meta["custom"]["module_fqn"]``.

    Call once after model construction and before tracing/compilation.
    Required for ``transformer_block_bucketing`` to identify per-layer nodes.
    """
    for fqn, submodule in model.named_modules():
        if fqn:
            submodule.forward = annotate_fn({_MODULE_FQN: fqn})(submodule.forward)


def remove_detach_pass(gm: torch.fx.GraphModule, example_inputs=None) -> torch.fx.GraphModule:
    """Remove ``aten.detach.default`` nodes — no-ops in a fully-traced graph."""
    count = 0
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target is torch.ops.aten.detach.default:
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
            count += 1
    if count > 0:
        gm.graph.lint()
        gm.recompile()
    return gm


_IDENTITY_VIEW_TARGETS = {
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
}


def remove_identity_view_pass(gm: torch.fx.GraphModule, example_inputs=None) -> torch.fx.GraphModule:
    """Remove ``view`` / ``reshape`` / ``_unsafe_view`` nodes that are no-ops
    (output shape equals input shape)."""
    count = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in _IDENTITY_VIEW_TARGETS:
            continue
        inp = node.args[0]
        inp_val = inp.meta.get("val") if isinstance(inp, torch.fx.Node) else None
        out_val = node.meta.get("val")
        if inp_val is None or out_val is None:
            continue
        if not isinstance(inp_val, torch.Tensor) or not isinstance(out_val, torch.Tensor):
            continue
        if inp_val.shape == out_val.shape:
            node.replace_all_uses_with(inp)
            gm.graph.erase_node(node)
            count += 1
    if count > 0:
        gm.graph.lint()
        gm.recompile()
    return gm


def remove_identity_slice_pass(gm: torch.fx.GraphModule, example_inputs=None) -> torch.fx.GraphModule:
    """Remove ``aten.slice.Tensor`` nodes that select the full dimension
    (start=0, end>=dim_size, step=1)."""
    count = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not torch.ops.aten.slice.Tensor:
            continue
        args = node.args
        input_node = args[0]
        dim = args[1] if len(args) > 1 else 0
        start = args[2] if len(args) > 2 else 0
        end = args[3] if len(args) > 3 else sys.maxsize
        step = args[4] if len(args) > 4 else 1
        if start != 0 or step != 1:
            continue
        val = input_node.meta.get("val")
        if val is None:
            continue
        if end >= val.shape[dim]:
            node.replace_all_uses_with(input_node)
            gm.graph.erase_node(node)
            count += 1
    gm.graph.lint()
    gm.recompile()
    return gm


def normalize_view_ops_as_reshape(
    gm: torch.fx.GraphModule,
    example_inputs=None,
) -> torch.fx.GraphModule:
    """Replace ``aten.view`` and ``aten._unsafe_view`` with ``aten.reshape``.

    Downstream pattern-matching passes expect ``aten.reshape.default``.
    """
    view_targets = {torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in view_targets:
            node.target = torch.ops.aten.reshape.default
    gm.graph.lint()
    gm.recompile()
    return gm


def async_tensor_parallel_pass(gm: torch.fx.GraphModule, example_inputs: tuple) -> torch.fx.GraphModule:
    """Pipeline TP collectives with matmuls via symmetric memory.

    Fuses all-gather + matmul into ``symm_mem.fused_all_gather_matmul`` and
    matmul + reduce-scatter into ``symm_mem.fused_matmul_reduce_scatter``.
    """
    import warnings

    from torch._inductor.fx_passes.micro_pipeline_tp import micro_pipeline_tp_pass
    from torch._inductor.fx_passes.overlap_scheduling import get_group_name
    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

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


def create_extra_fsdp_pg(fsdp_mesh: DeviceMesh) -> None:
    """Create a second NCCL PG mirroring ``fsdp_mesh`` on a separate CUDA stream.

    Having a distinct communicator lets the runtime schedule all-gathers on a
    stream that is independent of reduce-scatters, enabling AG/RS overlap
    in the backward pass.  Use ``reassign_to_pg_pass`` afterward to route
    AG nodes to this new PG.

    Args:
        fsdp_mesh: The DeviceMesh representing the FSDP process group.
    """
    import torch.distributed as dist

    fsdp_pg = fsdp_mesh.get_group()
    original_name = fsdp_pg.group_name

    if original_name in _EXTRA_FSDP_PG_REGISTRY:
        return

    ranks = dist.get_process_group_ranks(fsdp_pg)
    pg = dist.new_group(ranks=ranks, group_desc="fsdp_extra", use_local_synchronization=True)
    _EXTRA_FSDP_PG_REGISTRY[original_name] = pg.group_name


def get_extra_fsdp_pg_name(original_pg_name: str) -> str | None:
    """Return the extra PG name for a given original FSDP PG name, or None."""
    return _EXTRA_FSDP_PG_REGISTRY.get(original_pg_name)


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


class JointManualOverlapScheduler(ManualOverlapScheduler):
    """Manual overlap scheduler for joint forward+backward graphs.

    For the aot_fx_trace path we trace a joint fwd+bwd graph and want to
    bucket and reorder both directions in a single pass.  This subclass
    produces the same bucketing and prefetch pattern as invoking the upstream
    ``manual_overlap_bucketing`` twice (once per direction).

    Overrides ``_manual_bucket_collectives`` to split each module's
    collectives by direction before handing them to the bucketer.

    Overrides ``_manual_reorder_graph`` to track per-direction state so a
    single reversed walk emits correct AG prefetch edges for both regions.
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        module_bucket_plans: list[list[str] | str],
        insert_overlap_deps: bool,
        *,
        is_backward_fn: Callable[[torch.fx.Node], bool],
        module_stack_fn: Callable[[torch.fx.Node], list[tuple[str, type[Any]]]],
        bucket_mode: BucketMode | None = None,
    ) -> None:
        super().__init__(
            gm,
            module_bucket_plans,
            insert_overlap_deps,
            module_stack_fn=module_stack_fn,
            bucket_mode=bucket_mode,
        )
        self._is_backward_fn = is_backward_fn

    def _manual_bucket_collectives(self) -> None:
        """Bucket per module, splitting by direction to keep fwd/bwd buckets disjoint."""
        self._obtain_nodes_in_subgraph()
        for nodes in self.nodes_in_subgraph:
            fwd_nodes = [n for n in nodes if not self._is_backward_fn(n)]
            bwd_nodes = [n for n in nodes if self._is_backward_fn(n)]
            if fwd_nodes:
                self.bucketer.manual_bucket_collectives(nodes=fwd_nodes)
            if bwd_nodes:
                self.bucketer.manual_bucket_collectives(nodes=bwd_nodes)

        _stable_topological_sort(self.graph, {})
        self.graph.lint()
        self.nodes = list(self.graph.nodes)
        self.in_degree = Counter(user for node in self.nodes for user in node.users)

    def _manual_reorder_graph(self) -> None:
        """Reorder with separate fwd/bwd buffers so AG pairing never crosses
        the fwd/bwd boundary. RS pairing is unchanged — RSs only occur in
        backward and are already direction-scoped."""
        overlap_deps: dict[torch.fx.Node, OrderedSet[torch.fx.Node]] = defaultdict(OrderedSet)

        self._schedule_rs_prefetch(overlap_deps)
        self._schedule_ag_prefetch(overlap_deps)

        _stable_topological_sort(self.graph, overlap_deps)
        self.graph.lint()

        if self.insert_overlap_deps:
            from torch._inductor.fx_passes.control_dependencies import preserve_node_ordering

            preserve_node_ordering(self.graph, overlap_deps)

    def _schedule_rs_prefetch(
        self,
        overlap_deps: dict[torch.fx.Node, OrderedSet[torch.fx.Node]],
    ) -> None:
        """Top-down loop that emits RS prefetch edges and populates ``self.scheduled``."""
        delayed_rs_wait_nodes: list[torch.fx.Node] = []
        current_rs_start_nodes: list[torch.fx.Node] = []

        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self.on_path_ready = []
        self.scheduled = OrderedSet()
        for node in self.nodes:
            if self.in_degree[node] == 0:
                self._add_to_ready_queue(node)

        while self.on_path_ready:
            _, node = heapq.heappop(self.on_path_ready)
            node_type = self.bucketer.bucketed_node_types.get(node, "")

            if node in self.scheduled:
                continue

            if node_type == "bucketed_reduce_scatter":
                current_rs_start_nodes.append(node)
            elif node_type == "bucketed_reduce_scatter_wait":
                if current_rs_start_nodes:
                    for delayed in delayed_rs_wait_nodes:
                        for rs_start in current_rs_start_nodes:
                            overlap_deps[delayed].add(rs_start)
                    delayed_rs_wait_nodes.clear()
                    current_rs_start_nodes.clear()
                delayed_rs_wait_nodes.append(node)

            self._schedule(node)

    def _schedule_ag_prefetch(
        self,
        overlap_deps: dict[torch.fx.Node, OrderedSet[torch.fx.Node]],
    ) -> None:
        """Reversed walk that emits per-direction AG prefetch edges.

        Uses separate fwd/bwd buffers so AG pairing never crosses the fwd/bwd
        boundary. Consumes ``self.scheduled`` produced by ``_schedule_rs_prefetch``.
        """
        self.scheduled = OrderedSet(reversed(list(self.scheduled)))

        bwd_scope: OrderedSet[torch.fx.Node] = OrderedSet()
        fwd_scope: OrderedSet[torch.fx.Node] = OrderedSet()
        for sublist in self.nodes_in_subgraph:
            for n in sublist:
                if self._is_backward_fn(n):
                    bwd_scope.add(n)
                else:
                    fwd_scope.add(n)

        bwd_picked: list[torch.fx.Node] = []
        fwd_picked: list[torch.fx.Node] = []
        bwd_last_compute: torch.fx.Node | None = None
        fwd_last_compute: torch.fx.Node | None = None

        for node in self.scheduled:
            node_type = self.bucketer.bucketed_node_types.get(node, "")
            is_bwd = self._is_backward_fn(node)
            picked = bwd_picked if is_bwd else fwd_picked

            if node_type == "bucketed_all_gather":
                picked.append(node)
                continue

            if node_type == "bucketed_all_gather_wait":
                if picked:
                    for ag in picked:
                        overlap_deps[self.bucketer.node_to_wait_map[node]].add(ag)
                picked.clear()

            if is_compute_node(node):
                if is_bwd and node in bwd_scope:
                    bwd_last_compute = node
                elif not is_bwd and node in fwd_scope:
                    fwd_last_compute = node

        self._apply_trailing_block(bwd_picked, bwd_last_compute, overlap_deps)
        self._apply_trailing_block(fwd_picked, fwd_last_compute, overlap_deps)

    def _apply_trailing_block(
        self,
        picked: list[torch.fx.Node],
        last_compute: torch.fx.Node | None,
        overlap_deps: dict[torch.fx.Node, OrderedSet[torch.fx.Node]],
    ) -> None:
        if last_compute is None or not picked:
            return
        if OrderedSet(picked) & OrderedSet(self.node_ancestors[last_compute]):
            return
        for ag in picked:
            overlap_deps[last_compute].add(ag)


def joint_transformer_block_bucketing_reordering_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    module_bucket_plans: list[list[str] | str],
    insert_overlap_deps: bool = False,
    bucket_mode: BucketMode | None = None,
) -> torch.fx.GraphModule:
    """Run joint-graph manual bucketing and reordering.

    Joint-graph equivalent of ``manual_overlap_bucketing`` that handles fwd
    all-gathers, bwd all-gathers, and bwd reduce-scatters in one pass over
    the joint graph.

    Requires ``annotate_module_fqns`` to have been called before tracing so
    that nodes carry ``module_fqn`` in ``node.meta["custom"]``.
    """

    def _is_backward(node: torch.fx.Node) -> bool:
        return _is_backward_node(node) or _is_recomputed_node(node)

    def _stack_fn(node: torch.fx.Node) -> list[tuple[str, type]]:
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN)
        if not fqn:
            return []
        return [(fqn, torch.nn.Module)]

    overlapped_gm = JointManualOverlapScheduler(
        gm,
        module_bucket_plans,
        insert_overlap_deps,
        is_backward_fn=_is_backward,
        module_stack_fn=_stack_fn,
        bucket_mode=bucket_mode,
    ).run()
    overlapped_gm.recompile()
    return overlapped_gm


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
        import functools

        _tb_pass = functools.partial(
            transformer_block_bucketing_reordering_pass,
            fsdp_manual_buckets=fsdp_manual_buckets,
        )
        if backend == "aot_eager":
            inner_backend = aot_autograd_backend(
                fw_compiler=_tb_pass,
                bw_compiler=_tb_pass,
                keep_inference_input_mutations=True,
            )
        elif backend in ("inductor", "neuron"):

            def _inductor_tb_pass(gm: torch.fx.Graph) -> torch.fx.GraphModule:
                from torch._inductor.fx_passes.overlap_manual_scheduling import manual_overlap_bucketing as _mub

                return _mub(
                    gm.owning_module,
                    module_bucket_plans=fsdp_manual_buckets,
                    insert_overlap_deps=True,
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
