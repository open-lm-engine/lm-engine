# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import torch

from ..parallel import ProcessGroupManager
from .rotaters import recv, send


class _SequencePipeline(torch.autograd.Function):
    """Runs a stateful sequence op as a pipeline over the context parallel ranks.

    A non-linear recurrence has no associative operator to scan over, so the chunk boundaries carry
    a true serial dependency: rank `r` cannot start until rank `r - 1` has produced the state at its
    boundary. Forward therefore walks the ranks left to right and backward walks them right to left,
    which makes this pipeline parallelism along the sequence dimension.

    The hand-off is driven explicitly here instead of by autograd-aware send/recv primitives, and the
    ordering constraints are strict enough to be worth spelling out. Every rank has to agree on the
    order of the messages, so all communication has to sit at points in the schedule that do not
    depend on how the autograd engine or an enclosing activation checkpoint happens to interleave
    things:

    - a rank's receive and send both live inside this one node, so the engine cannot slot one block's
      send between another block's receive and its matching send. With the two halves in separate
      nodes the engine orders them by its own heuristics, and those heuristics see a different graph
      on the first and last rank (neither has both halves) so the ranks can disagree.

    - backward recomputes the local op instead of holding on to its graph. Holding the graph means
      holding saved tensors that an enclosing activation checkpoint owns, and unpacking those inside
      backward() fires that checkpoint's recompute from inside this node -- issuing this rank's
      communication at a point in the schedule no other rank shares. The recompute needs no
      communication of its own precisely because forward saved the boundary state it starts from.

    The inputs and the boundary state are held as plain attributes on ctx rather than through
    save_for_backward, for the same reason: save_for_backward would hand them to an enclosing
    checkpoint and put the recompute back inside backward().
    """

    @staticmethod
    def forward(
        ctx, function: Callable, state_shape: tuple[int, ...], *tensors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_rank = ProcessGroupManager.get_context_parallel_rank()
        cp_world_size = ProcessGroupManager.get_context_parallel_world_size()

        is_first_rank = cp_rank == 0
        is_last_rank = cp_rank == cp_world_size - 1

        # everything crossing the wire is cast to this, so both ends always agree on the layout
        state_dtype = tensors[0].dtype
        state_device = tensors[0].device

        input_state = None
        if not is_first_rank:
            input_state = torch.empty(state_shape, dtype=state_dtype, device=state_device)
            recv(input_state)

        with torch.no_grad():
            x, output_state = function(*tensors, input_state)

        if not is_last_rank:
            send(output_state.to(state_dtype))

        ctx.function = function
        ctx.function_inputs = tensors
        ctx.input_state = input_state
        ctx.state_shape = state_shape
        ctx.state_dtype = state_dtype
        ctx.state_device = state_device
        ctx.is_first_rank = is_first_rank
        ctx.is_last_rank = is_last_rank
        ctx.set_materialize_grads(False)

        return x, output_state

    @staticmethod
    def backward(ctx, dx: torch.Tensor | None, d_output_state: torch.Tensor | None) -> tuple:
        if not ctx.is_last_rank:
            # the next rank seeded its chunk with our boundary state, so that state's gradient is
            # only known once the next rank has finished its own backward
            grad_from_next_rank = torch.empty(ctx.state_shape, dtype=ctx.state_dtype, device=ctx.state_device)
            recv(grad_from_next_rank, shift=-1)

            d_output_state = grad_from_next_rank if d_output_state is None else d_output_state + grad_from_next_rank

        # purely local, no communication: forward saved the state this chunk starts from
        tensors = tuple(t.detach().requires_grad_(t.requires_grad) for t in ctx.function_inputs)
        input_state = None if ctx.input_state is None else ctx.input_state.detach().requires_grad_(True)

        with torch.enable_grad():
            x, output_state = ctx.function(*tensors, input_state)

        outputs = []
        output_grads = []

        for output, output_grad in ((x, dx), (output_state, d_output_state)):
            if output_grad is not None:
                outputs.append(output)
                output_grads.append(output_grad)

        inputs = [tensor for tensor in tensors if tensor.requires_grad]
        if input_state is not None:
            inputs.append(input_state)

        grads = list(torch.autograd.grad(outputs=outputs, grad_outputs=output_grads, inputs=inputs, allow_unused=True))

        if not ctx.is_first_rank:
            input_state_grad = grads.pop()

            # the previous rank is blocked on this message, so it has to be sent even when the op
            # turns out not to use its input state
            if input_state_grad is None:
                input_state_grad = torch.zeros(ctx.state_shape, dtype=ctx.state_dtype, device=ctx.state_device)

            send(input_state_grad.to(ctx.state_dtype), shift=-1)

        grads = iter(grads)
        input_grads = tuple(next(grads) if tensor.requires_grad else None for tensor in ctx.function_inputs)

        return None, None, *input_grads


@torch._dynamo.disable
def sequence_pipeline(
    function: Callable, tensors: tuple[torch.Tensor, ...], state_shape: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pipeline `function` over the context parallel ranks along the sequence dimension.

    `function` is called as `function(*tensors, input_state)` and must return `(output, state)`.
    `input_state` is `None` on the first rank and the previous rank's boundary state elsewhere;
    `state_shape` is the shape of that state. Gradients flow back through the whole pipeline, so
    this is exact rather than a truncation at the chunk boundaries.

    `function` is run once in forward and once more in backward, so it has to be deterministic.
    """

    return _SequencePipeline.apply(function, state_shape, *tensors)
