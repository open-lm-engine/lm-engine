# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from parameterized import parameterized

from lm_engine.hf_models.modeling_utils import get_activation_function
from lm_engine.hf_models.modeling_utils.activations import get_base_activation
from lm_engine.hf_models.modeling_utils.activations.glu import GLUActivation

from ..test_common import TestCommons


class ActivationsTest(TestCommons):
    @parameterized.expand(TestCommons.make_args_matrix(TestCommons.get_all_devices()))
    def test_sigmoid_glu(self, device: torch.device) -> None:
        self.skip_test_if_device_unavailable(device)

        base_sigmoid = get_base_activation("sigmoid")
        sigmoid_glu = GLUActivation(base_sigmoid)

        pytorch_glu = get_activation_function("glu")

        x = torch.randn(10, 10, device=device)
        sigmoid_glu_output = sigmoid_glu(x)
        pytorch_glu_output = pytorch_glu(x)

        self.assert_equal_tensors(sigmoid_glu_output, pytorch_glu_output, True)
