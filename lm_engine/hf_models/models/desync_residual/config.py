# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...config import CommonConfig


class DesyncResidualConfig(CommonConfig):
    model_type = "desync_residual"

    def __init__(
        self,
        pretraining_tensor_parallel_size: int = 1,
        reduce_pattern: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.pretraining_tensor_parallel_size = pretraining_tensor_parallel_size

        self.reduce_pattern = (
            [{"attention": True, "mlp": True} for i in range(self.num_layers)]
            if reduce_pattern is None
            else reduce_pattern
        )
