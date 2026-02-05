# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from ...dtensors import modify_state_dict_to_dtensor_dict
from ...utils import ProcessGroupManager


class DTensorModule(nn.Module):
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        if ProcessGroupManager.is_tensor_parallel_enabled():
            state_dict = modify_state_dict_to_dtensor_dict(self, state_dict=state_dict, prefix="", strip_keys=False)

        super().load_state_dict(state_dict, strict, assign)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        if ProcessGroupManager.is_tensor_parallel_enabled():
            state_dict = modify_state_dict_to_dtensor_dict(self, state_dict=state_dict, prefix=prefix, strip_keys=True)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
