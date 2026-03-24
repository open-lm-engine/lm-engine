# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import itertools

import pytest
import torch

from lm_engine.enums import Kernel
from lm_engine.kernels import enable_kernels
from lm_engine.utils import is_flash_attention_2_available, is_flash_attention_3_available, set_seed

from ..test_common import (
    assert_equal_tensors,
    from_config,
    get_dense_test_config,
    get_dummy_inputs,
    skip_test_if_device_unavailable,
)


SEED = 1234


@pytest.mark.parametrize("device", [torch.device("cuda")])
@pytest.mark.parametrize("position_embedding_type", ["learned_absolute", "rope"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdpa_padding_free_transformer_equivalence(
    device: torch.device, position_embedding_type: str, dtype: torch.dtype
) -> None:
    skip_test_if_device_unavailable(device)

    kernel = None
    if is_flash_attention_3_available():
        kernel = Kernel.flash_attention_3
    if is_flash_attention_2_available():
        kernel = Kernel.flash_attention_2

    if kernel is None:
        pytest.skip("skipping test because flash attention 2 or 3 is unavailable")

    set_seed(SEED)

    config = get_dense_test_config(position_embedding_type, num_layers=1)

    sdpa_model = from_config(config, dtype=dtype).to(device)
    flash_model = from_config(config, dtype=dtype, use_padding_free_transformer=True).to(device)

    sdpa_model.eval()
    flash_model.eval()

    flash_model.load_state_dict(sdpa_model.state_dict())

    input_ids, attention_mask, labels = get_dummy_inputs(device)
    sdpa_output = sdpa_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    attention_mask = attention_mask.to(torch.bool)
    sdpa_logits = sdpa_output.logits
    sdpa_logits = torch.cat([sdpa_logits[i, ex, :] for i, ex in enumerate(attention_mask)])
    sdpa_loss = sdpa_output.loss

    with enable_kernels([kernel]):
        input_ids, attention_mask, labels = get_dummy_inputs(device, return_list=True)
        flash_output = flash_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        flash_logits = flash_output.logits
        flash_loss = flash_output.loss

    assert_equal_tensors(
        sdpa_logits,
        flash_logits,
        False,
        rtol_float16=1e-3,
        atol_float16=3e-4,
        rtol_bfloat16=5e-3,
        atol_bfloat16=5e-3,
    )

    assert_equal_tensors(sdpa_loss, flash_loss, False, atol_float32=1.2e-4, rtol_float32=0)


@pytest.mark.parametrize("device", [torch.device("cuda")])
@pytest.mark.parametrize("position_embedding_type", ["learned_absolute", "rope"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdpa_flash_attention_equivalence(
    device: torch.device, position_embedding_type: str, dtype: torch.dtype
) -> None:
    skip_test_if_device_unavailable(device)

    kernel = None
    if is_flash_attention_3_available():
        kernel = Kernel.flash_attention_3
    if is_flash_attention_2_available():
        kernel = Kernel.flash_attention_2

    if kernel is None:
        pytest.skip("skipping test because flash attention 2 or 3 is unavailable")

    set_seed(SEED)

    input_ids, attention_mask, labels = get_dummy_inputs(device)
    config = get_dense_test_config(position_embedding_type, num_layers=1)

    model = from_config(config, dtype=dtype).to(device)
    model.eval()

    sdpa_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    sdpa_logits = sdpa_output.logits
    sdpa_loss = sdpa_output.loss

    with enable_kernels([kernel]):
        flash_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        flash_logits = flash_output.logits
        flash_loss = flash_output.loss

    # we don't care about what happens on masked values (they don't match btw)
    sdpa_logits[attention_mask == 0] = 0
    flash_logits[attention_mask == 0] = 0

    assert_equal_tensors(
        sdpa_logits[attention_mask],
        flash_logits[attention_mask],
        False,
        rtol_float16=1e-3,
        atol_float16=3e-4,
        rtol_bfloat16=5e-3,
        atol_bfloat16=5e-3,
    )

    assert_equal_tensors(sdpa_loss, flash_loss, False, atol_float32=1.2e-4, rtol_float32=0)


@pytest.mark.parametrize("device", [torch.device("cuda")])
@pytest.mark.parametrize("position_embedding_type", ["learned_absolute", "rope"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_padding_free_transformer_with_list_and_tensor(
    device: torch.device, position_embedding_type: str, dtype: torch.dtype
) -> None:
    skip_test_if_device_unavailable(device)

    kernel = None
    if is_flash_attention_3_available():
        kernel = Kernel.flash_attention_3
    if is_flash_attention_2_available():
        kernel = Kernel.flash_attention_2

    if kernel is None:
        pytest.skip("skipping test because flash attention 2 or 3 is unavailable")

    set_seed(SEED)

    config = get_dense_test_config(position_embedding_type, num_layers=1)

    model = from_config(config, dtype=dtype, use_padding_free_transformer=True).to(device)
    model.eval()

    with enable_kernels([kernel]):
        input_ids, attention_mask, labels = get_dummy_inputs(device, return_list=True)
        list_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        list_logits = list_output.logits
        list_loss = list_output.loss

        seqlens = torch.tensor([0] + [len(i) for i in input_ids])
        cu_seqlens = seqlens.cumsum(dim=-1).to(device, torch.int32)
        max_seqlen = seqlens.max().item()
        position_ids = torch.tensor(list(itertools.chain(*[list(range(len(i))) for i in input_ids])), device=device)
        input_ids = torch.tensor(list(itertools.chain(*input_ids)), device=device)
        labels = torch.tensor(list(itertools.chain(*labels)), device=device)
        tensor_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        tensor_logits = tensor_output.logits
        tensor_loss = tensor_output.loss

    assert_equal_tensors(list_logits, tensor_logits, True)
    assert_equal_tensors(list_loss, tensor_loss, True)


@pytest.mark.parametrize("device", [torch.device("cuda")])
@pytest.mark.parametrize("position_embedding_type", ["learned_absolute", "rope"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdpa_flash_enabled(device: torch.device, position_embedding_type: str, dtype: torch.dtype) -> None:
    skip_test_if_device_unavailable(device)

    set_seed(SEED)

    config = get_dense_test_config(position_embedding_type, num_layers=1)

    model = from_config(config, dtype=dtype).to(device)
    model.eval()

    input_ids, _, labels = get_dummy_inputs(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.int, device=device)

    sdpa_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    sdpa_logits = sdpa_output.logits
    sdpa_loss = sdpa_output.loss

    flash_output = model(input_ids=input_ids, labels=labels)
    flash_logits = flash_output.logits
    flash_loss = flash_output.loss

    assert_equal_tensors(
        sdpa_logits,
        flash_logits,
        False,
        rtol_float16=1e-3,
        atol_float16=3e-4,
        rtol_bfloat16=5e-3,
        atol_bfloat16=5e-3,
    )

    assert_equal_tensors(sdpa_loss, flash_loss, False, atol_float32=3.8e-4, rtol_float32=0)


@pytest.mark.parametrize("device", [torch.device("cuda")])
@pytest.mark.parametrize("position_embedding_type", ["learned_absolute", "rope"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attention_equivalence_with_and_without_attention_masks(
    device: torch.device, position_embedding_type: str, dtype: torch.dtype
) -> None:
    skip_test_if_device_unavailable(device)

    kernel = None
    if is_flash_attention_3_available():
        kernel = Kernel.flash_attention_3
    if is_flash_attention_2_available():
        kernel = Kernel.flash_attention_2

    if kernel is None:
        pytest.skip("skipping test because flash attention 2 or 3 is unavailable")

    set_seed(SEED)

    input_ids, _, labels = get_dummy_inputs(device)
    config = get_dense_test_config(position_embedding_type, num_layers=1)

    attention_mask = torch.ones_like(input_ids)

    model = from_config(config, dtype=dtype).to(device)
    model.eval()

    with enable_kernels([kernel]):
        output_with_mask = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits_with_mask = output_with_mask.logits
        loss_with_mask = output_with_mask.loss

        output_without_mask = model(input_ids=input_ids, labels=labels)
        logits_without_mask = output_without_mask.logits
        loss_without_mask = output_without_mask.loss

    assert_equal_tensors(logits_with_mask, logits_without_mask, True)
    assert_equal_tensors(loss_with_mask, loss_without_mask, True)
