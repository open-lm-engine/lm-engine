# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import pytest
import torch

from lm_engine.hf_models.modeling_utils import get_activation_function
from lm_engine.hf_models.modeling_utils.activations import get_base_activation
from lm_engine.hf_models.modeling_utils.activations.glu import GLUActivation

from ...test_common import skip_test_if_device_unavailable
from ..test_common import assert_equal_tensors


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("is_interleaved", [False, True])
def test_sigmoid_glu(device: torch.device, is_interleaved: bool) -> None:
    skip_test_if_device_unavailable(device)

    base_sigmoid = get_base_activation("sigmoid")
    sigmoid_glu = GLUActivation(base_sigmoid)

    pytorch_glu = get_activation_function("glu")

    x = torch.randn(10, 10, device=device)
    sigmoid_glu_output = sigmoid_glu(x, is_interleaved=is_interleaved)
    pytorch_glu_output = pytorch_glu(x, is_interleaved=is_interleaved)

    assert_equal_tensors(sigmoid_glu_output, pytorch_glu_output, True)
