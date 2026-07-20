# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from dataclasses import dataclass

import torch

from ...accelerator import Accelerator
from ...parallel import ProcessGroupManager, prepare_context_parallel_input


@dataclass
class PositionInfo:
    position_ids: torch.Tensor | None = None
    rope_cos_sin: torch.Tensor | None = None

    def reset_parameters(
        self,
        attention_mask: torch.Tensor | None,
        past_length: int,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> None:
        assert self.position_ids is None

        if attention_mask is not None and attention_mask.dim() == 2:
            assert not ProcessGroupManager.is_context_parallel_enabled()
            # create position_ids on the fly for batch generation
            position_ids = (
                attention_mask.to(
                    torch.int32 if Accelerator.get_accelerator() == Accelerator.trainium else torch.int64
                ).cumsum(-1)
                - 1
            )

            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_length > 0:
                position_ids = position_ids[:, past_length:key_length:]
        else:
            position_ids = torch.arange(
                past_length,
                key_length,
                dtype=torch.int32 if Accelerator.get_accelerator() == Accelerator.trainium else torch.int64,
                device=device,
            )

            position_ids = position_ids.unsqueeze(0).view(-1, query_length)

        self.position_ids = prepare_context_parallel_input(inputs=(position_ids,))[0]
