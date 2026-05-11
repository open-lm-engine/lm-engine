# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from .compile import (
    annotate_module_fqns,
    async_tensor_parallel_pass,
    autobucketing_reordering_pass,
    create_extra_fsdp_pg,
    get_extra_fsdp_pg_name,
    get_simple_fsdp_compile_backend,
    normalize_view_ops_as_reshape,
    reassign_to_pg_pass,
    remove_detach_pass,
    remove_identity_slice_pass,
    remove_identity_view_pass,
    transformer_block_bucketing_reordering_pass,
)
from .fsdp import MixedPrecisionPolicy, data_parallel
