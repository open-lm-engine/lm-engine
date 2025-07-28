# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ..containers import BackwardHookOptimizerContainer, ModelContainer, OptimizerContainer
from .optimizer import get_optimizer_container
from .scheduler import get_scheduler_container


def get_learning_rate(model_container: ModelContainer, lr_scheduler_container: OptimizerContainer) -> float:
    if isinstance(lr_scheduler_container, BackwardHookOptimizerContainer):
        parameters = model_container[0].parameters()
        param = next(parameters)

        lr = param._lr_scheduler.get_last_lr()[0]
    else:
        lr = lr_scheduler_container[0].get_last_lr()[0]

    return lr
