# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Any

import torch.distributed
from torch.distributed import ProcessGroup

from .utils import ProcessGroupManager


class Communication:
    @staticmethod
    def broadcast_object(obj: Any, src: int, group: ProcessGroup) -> Any:
        if ProcessGroupManager.get_global_rank() != src:
            obj = None

        object_list = [obj]
        torch.distributed.broadcast_object_list(object_list, src=src, group=group)
        obj = object_list[0]

        return obj
