# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from torch.testing import assert_close

from lm_engine.modeling_utils.sequence_mixer_blocks.mamba2.op import mamba2_torch


def test_mamba2_torch_chunk_matches_recurrent_for_grouped_heads() -> None:
    torch.manual_seed(0)

    batch_size, sequence_length = 2, 7
    num_heads, head_dim, state_size, num_groups = 4, 3, 5, 2
    chunk_size = 4

    x = torch.randn(batch_size, sequence_length, num_heads * head_dim)
    dt = torch.rand(batch_size, sequence_length, num_heads)
    A_neg = -torch.rand(num_heads)
    B = torch.randn(batch_size, sequence_length, num_groups, state_size)
    C = torch.randn(batch_size, sequence_length, num_groups, state_size)
    D = torch.randn(num_heads)

    chunk_output, chunk_state = mamba2_torch(
        x=x,
        dt=dt,
        A_neg=A_neg,
        B=B,
        C=C,
        D=D,
        h=None,
        use_recurrent=False,
        chunk_size=chunk_size,
    )

    recurrent_outputs = []
    recurrent_state = None
    for index in range(sequence_length):
        output, recurrent_state = mamba2_torch(
            x=x[:, index : index + 1],
            dt=dt[:, index : index + 1],
            A_neg=A_neg,
            B=B[:, index : index + 1],
            C=C[:, index : index + 1],
            D=D,
            h=recurrent_state,
            use_recurrent=True,
            chunk_size=chunk_size,
        )
        recurrent_outputs.append(output)

    recurrent_output = torch.cat(recurrent_outputs, dim=1)

    assert_close(chunk_output, recurrent_output, rtol=1e-5, atol=1e-5)
    assert_close(chunk_state, recurrent_state, rtol=1e-5, atol=1e-5)
