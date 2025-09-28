# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from lm_engine.hf_models.tensor import PackedTensor


y = torch.randn(5, 4, requires_grad=True)
# Example usage
x = PackedTensor.from_unpacked_tensor(y, batch_size=5)
x.sum().backward()

print(x)
