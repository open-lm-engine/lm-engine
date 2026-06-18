# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch

from lm_engine.hf_models.config import _Mamba2Args

from ...utils import get_dense_test_config, get_moe_test_config, model_conversion_test


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("add_bias", [False, True])
def test_llama_model_conversion(device: torch.device, add_bias: bool) -> None:
    lm_engine_config = get_dense_test_config(
        "rope", add_bias=add_bias, activation_function="swiglu", normalization_function="rmsnorm"
    )

    model_conversion_test(lm_engine_config=lm_engine_config, model_type="llama", device=device, exact_match=False)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("add_bias", [False, True])
def test_granite_model_conversion(device: torch.device, add_bias: bool) -> None:
    lm_engine_config = get_dense_test_config(
        "rope",
        add_bias=add_bias,
        activation_function="swiglu",
        normalization_function="rmsnorm",
        m_emb=2,
        m_width=2,
    )

    model_conversion_test(lm_engine_config=lm_engine_config, model_type="granite", device=device, exact_match=False)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("is_moe", [False, True])
def test_granitemoehybrid_model_conversion(device: torch.device, is_moe: bool) -> None:
    if is_moe:
        lm_engine_config = get_moe_test_config(
            "nope",
            add_bias=False,
            shared_n_inner=64,
            activation_function="swiglu",
            normalization_function="rmsnorm",
            m_emb=2,
            m_width=2,
        )
    else:
        lm_engine_config = get_dense_test_config(
            "nope",
            add_bias=False,
            activation_function="swiglu",
            normalization_function="rmsnorm",
            m_emb=2,
            m_width=2,
        )

    for layer in range(lm_engine_config.num_layers):
        if layer % 2 == 0:
            lm_engine_config.sequence_mixer_blocks[layer] = _Mamba2Args(intermediate_size=256)

    model_conversion_test(
        lm_engine_config=lm_engine_config,
        model_type="granitemoehybrid",
        device=device,
        exact_match=False,
        compare_loss=False,
        logits_atol_float32=2.5e-5,
    )
