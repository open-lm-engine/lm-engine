# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F
from parameterized import parameterized
from transformers import set_seed

from lm_engine.hf_models.models.desync_residual.linear import DesyncResidualLinear

from ..test_common import TestCommons


SEED = 1234


class DesyncResidualLinearTest(TestCommons):
    @parameterized.expand([False, True])
    def test_desync_residual_linear_correctness(self, bias: bool) -> None:
        set_seed(SEED)

        tp_world_size = 8
        in_features = 7
        out_features = 11

        desync_residual_linear = DesyncResidualLinear(in_features, out_features, tp_world_size, std=0.02, bias=bias)
        input = torch.randn(tp_world_size, 4, 37, in_features)

        if bias:
            with torch.no_grad():
                nn.init.normal_(desync_residual_linear.bias, mean=0, std=desync_residual_linear.std)

        output = []
        for rank in range(tp_world_size):
            i = input[rank, ...]
            w = desync_residual_linear.weight.view(tp_world_size, in_features, out_features)[rank, :, :].T
            b = (
                None
                if desync_residual_linear.bias is None
                else desync_residual_linear.bias.view(tp_world_size, -1)[rank, :]
            )

            output.append(F.linear(i, w, b))

        output = torch.stack(output, dim=0)
        desync_residual_output = desync_residual_linear(input)

        assert output.equal(desync_residual_output)

    @parameterized.expand([False, True])
    def test_desync_residual_linear_correctness_with_broadcast_input(self, bias: bool) -> None:
        set_seed(SEED)

        tp_world_size = 8
        in_features = 7
        out_features = 11

        desync_residual_linear = DesyncResidualLinear(in_features, out_features, tp_world_size, std=0.02, bias=bias)
        input = torch.randn(1, 4, 37, in_features)

        if bias:
            with torch.no_grad():
                nn.init.normal_(desync_residual_linear.bias, mean=0, std=desync_residual_linear.std)

        output = []
        for rank in range(tp_world_size):
            w = desync_residual_linear.weight.view(tp_world_size, in_features, out_features)[rank, ...].T
            b = (
                None
                if desync_residual_linear.bias is None
                else desync_residual_linear.bias.view(tp_world_size, -1)[rank, :]
            )

            output.append(F.linear(input.squeeze(0), w, b))

        output = torch.stack(output, dim=0)
        desync_residual_output = desync_residual_linear(input)

        assert output.equal(desync_residual_output)
