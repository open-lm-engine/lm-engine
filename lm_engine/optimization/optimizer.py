# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging

import torch.nn as nn
from torch.optim import ASGD as TorchASGD
from torch.optim import LBFGS as TorchLBFGS
from torch.optim import SGD as TorchSGD
from torch.optim import Adadelta as TorchAdadelta
from torch.optim import Adagrad as TorchAdagrad
from torch.optim import Adam as TorchAdam
from torch.optim import Adamax as TorchAdamax
from torch.optim import AdamW as TorchAdamW
from torch.optim import NAdam as TorchNAdam
from torch.optim import Optimizer
from torch.optim import RAdam as TorchRAdam
from torch.optim import RMSprop as TorchRMSprop
from torch.optim import Rprop as TorchRprop

from ..containers import BackwardHookOptimizerContainer, ModelContainer, OptimizerContainer
from ..enums import ParamsGroupMethod
from ..hf_models import get_optimizer_split_function
from ..utils import log_rank_0
from .params_group import _ParamsGroupsList, get_param_groups_list
from .split_param_optimizer import SplitParamOptimizer


# https://pytorch.org/docs/stable/optim.html
_OPTIMIZER_CLASSES = {
    "TorchAdadelta": TorchAdadelta,
    "TorchAdagrad": TorchAdagrad,
    "TorchAdam": TorchAdam,
    "TorchAdamax": TorchAdamax,
    "TorchAdamW": TorchAdamW,
    "TorchASGD": TorchASGD,
    "TorchLBFGS": TorchLBFGS,
    "TorchNAdam": TorchNAdam,
    "TorchRAdam": TorchRAdam,
    "TorchRMSprop": TorchRMSprop,
    "TorchRprop": TorchRprop,
    "TorchSGD": TorchSGD,
}


def _build_optimizer(
    optimizer_class, params_groups: _ParamsGroupsList, optimizer_class_args: dict, split_params_for_optimizer: bool
) -> SplitParamOptimizer | Optimizer:
    if not split_params_for_optimizer:
        return optimizer_class(params_groups.to_torch_compatible_params_groups(), **optimizer_class_args)

    proxy_grad_fns: dict[int, tuple] = {}
    split_params: set[nn.Parameter] = set()
    modified_groups = []

    for pg in params_groups.params_groups:
        group = pg.to_param_group()
        names = pg.get_param_names()
        group_kwargs = {k: v for k, v in group.items() if k != "params"}
        new_params = []
        for param, name in zip(group["params"], names):
            split_fn = get_optimizer_split_function(param)
            if split_fn is None:
                new_params.append(param)
            else:
                log_rank_0(logging.INFO, f"splitting {name} for optimizer")
                pieces = split_fn(param.data)

                for i, piece in enumerate(pieces):
                    proxy = nn.Parameter(piece)
                    new_params.append(proxy)
                    proxy_grad_fns[id(proxy)] = (param, lambda g, fn=split_fn, idx=i: fn(g)[idx])
                split_params.add(param)

        modified_groups.append({"params": new_params, **group_kwargs})

    inner = optimizer_class(modified_groups, **optimizer_class_args)

    if split_params:
        inner = SplitParamOptimizer(inner=inner, proxy_grad_fns=proxy_grad_fns, split_params=split_params)

    return inner


def get_optimizer_container(
    optimizer_class_name: str,
    optimizer_class_args: dict,
    model_container: ModelContainer,
    params_group_method: ParamsGroupMethod,
    use_optimizer_with_backward_hook: bool,
    split_params_for_optimizer: bool,
) -> OptimizerContainer:
    """setup list of optimizers for the model

    Args:
        optimizer_class_name (str): optimizer class name
        optimizer_class_args (dict): args for the optimizer class
        model_container (ModelContainer): model container
        params_group_method (ParamsGroupMethod): the params grouping to use
        use_optimizer_with_backward_hook (bool): whether to use optimizer as a backward hook
        split_params_for_optimizer (bool): whether to split params using model-defined split functions

    Returns:
        OptimizerContainer: optimizer container
    """

    if optimizer_class_name not in _OPTIMIZER_CLASSES:
        raise ValueError(f"invalid class_name ({optimizer_class_name}) for optimizer")

    optimizer_class = _OPTIMIZER_CLASSES[optimizer_class_name]
    if optimizer_class is None:
        raise ImportError("relevant package for the optimizer is not installed")

    params_groups_list = get_param_groups_list(model_container, optimizer_class_args, params_group_method)

    if use_optimizer_with_backward_hook:
        assert not split_params_for_optimizer

        for model, params_groups in zip(model_container, params_groups_list):
            for param_name, param in model.named_parameters():
                for group in params_groups.params_groups:
                    if param_name in group.parameter_name_map:
                        param._optimizer = optimizer_class(
                            [{"params": [param], **group.params_group_kwargs}], **optimizer_class_args
                        )

                        def _step(p: nn.Parameter) -> None:
                            p._optimizer.step()
                            p._optimizer.zero_grad()

                        param.register_post_accumulate_grad_hook(_step)

                        break

        optimizer_list = BackwardHookOptimizerContainer([None] * len(model_container))
    else:
        optimizer_list = OptimizerContainer(
            [
                _build_optimizer(
                    optimizer_class=optimizer_class,
                    params_groups=params_groups,
                    optimizer_class_args=optimizer_class_args,
                    split_params_for_optimizer=split_params_for_optimizer,
                )
                for params_groups in params_groups_list
            ]
        )

    return optimizer_list
