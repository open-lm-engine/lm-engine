# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from ..utils import is_torch_xla_available
from .context_parallel import prepare_context_parallel_input
from .manager import ProcessGroupManager, run_rank_n
from .pipeline_parallel import get_pipeline_stage_ids_on_current_rank
from .tensor_parallel import broadcast_tensor_parallel_input


if not is_torch_xla_available():
    from .simple_fsdp import MixedPrecisionPolicy, data_parallel, get_simple_fsdp_compile_backend
