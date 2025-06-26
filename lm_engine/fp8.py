# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [Note] Getting the 'torchao' package:
# This script requires the 'torchao' package to function correctly.
# Please ensure you have this package installed from the appropriate repository.
# You can obtain it from https://github.com/pytorch/ao by following the
# installation instructions.

# Note: Performance
# Float8 experimental is intended to be ran under `torch.compile`` for competitive performance

from typing import Callable

import torch

from .containers import ModelContainer
from .utils import is_torchao_available


if is_torchao_available():
    from torchao.float8 import (
        CastConfig,
        Float8LinearConfig,
        ScalingType,
        convert_to_float8_training,
        precompute_float8_dynamic_scale_for_fsdp,
        sync_float8_amax_and_scale_history,
    )
    from torchao.float8.fsdp_utils import (
        WeightWithDelayedFloat8CastTensor,
        WeightWithDynamicFloat8CastTensor,
        WeightWithStaticFloat8CastTensor,
    )

    torch.serialization.add_safe_globals(
        [WeightWithDynamicFloat8CastTensor, WeightWithDelayedFloat8CastTensor, WeightWithStaticFloat8CastTensor]
    )

    _PRECOMPUTE_SCALE: bool = False
    _DELAYED_SCALING: bool = False
    _SYNC_FP8_AMAX_AND_SCALE_HISTORY: Callable = sync_float8_amax_and_scale_history

    class FP8Manager:
        def __init__(
            self,
            model_container: ModelContainer,
            enable_fsdp_fp8_all_gather: bool,
            precompute_fp8_dynamic_scale_for_fsdp: bool,
            torch_compile: bool,
            scaling_type_input: ScalingType,
            scaling_type_weight: ScalingType,
            scaling_type_grad_output: ScalingType,
        ) -> None:
            for model in model_container:
                convert_to_float8_training(
                    model,
                    config=Float8LinearConfig(
                        enable_fsdp_float8_all_gather=enable_fsdp_fp8_all_gather,
                        cast_config_input=CastConfig(scaling_type=scaling_type_input),
                        cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
                        cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
                        force_recompute_fp8_weight_in_bwd=True,
                    ),
                    module_filter_fn=lambda mod, fqn: fqn != "output",
                )

            global _PRECOMPUTE_SCALE, _DELAYED_SCALING, _SYNC_FP8_AMAX_AND_SCALE_HISTORY

            _PRECOMPUTE_SCALE = enable_fsdp_fp8_all_gather and precompute_fp8_dynamic_scale_for_fsdp
            _DELAYED_SCALING = (
                scaling_type_input is ScalingType.DELAYED
                or scaling_type_weight is ScalingType.DELAYED
                or scaling_type_grad_output is ScalingType.DELAYED
            )

            if torch_compile:
                _SYNC_FP8_AMAX_AND_SCALE_HISTORY = torch.compile(sync_float8_amax_and_scale_history)

        @staticmethod
        def precompute_float8_dynamic_scale_for_fsdp(model_container: ModelContainer) -> None:
            if not _PRECOMPUTE_SCALE:
                return

            for model in model_container:
                precompute_float8_dynamic_scale_for_fsdp(model)

        @staticmethod
        def sync_float8_amax_and_scale_history(model_container: ModelContainer) -> None:
            if not _DELAYED_SCALING:
                return

            for model in model_container:
                _SYNC_FP8_AMAX_AND_SCALE_HISTORY(model)
