# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch.nn as nn
from torch.optim import ASGD as TorchASGD
from torch.optim import LBFGS as TorchLBFGS
from torch.optim import SGD as TorchSGD
from torch.optim import Adadelta as TorchAdadelta
from torch.optim import Adagrad as TorchAdagrad
from torch.optim import Adam as TorchAdam
from torch.optim import Adamax as TorchAdamax
from torch.optim import AdamW as TorchAdamW
from torch.optim import Muon as TorchMuon
from torch.optim import NAdam as TorchNAdam
from torch.optim import RAdam as TorchRAdam
from torch.optim import RMSprop as TorchRMSprop
from torch.optim import Rprop as TorchRprop

from ..containers import BackwardHookOptimizerContainer, ModelContainer, OptimizerContainer
from ..enums import ParamsGroupMethod
from ..hf_models import get_optimizer_split_function
from .params_group import get_param_groups_list


# https://pytorch.org/docs/stable/optim.html
_OPTIMIZER_CLASSES = {
    "TorchAdadelta": TorchAdadelta,
    "TorchAdagrad": TorchAdagrad,
    "TorchAdam": TorchAdam,
    "TorchAdamax": TorchAdamax,
    "TorchAdamW": TorchAdamW,
    "TorchASGD": TorchASGD,
    "TorchLBFGS": TorchLBFGS,
    "TorchMuon": TorchMuon,
    "TorchNAdam": TorchNAdam,
    "TorchRAdam": TorchRAdam,
    "TorchRMSprop": TorchRMSprop,
    "TorchRprop": TorchRprop,
    "TorchSGD": TorchSGD,
}


_SPLIT_FUNCTION_INCOMPATIBLE_OPTIMIZERS = ["TorchMuon"]

# Parameter name substrings that must use AdamW instead of Muon (embeddings and lm_head)
_MUON_ADAMW_PARAM_NAMES = {"wte", "lm_head"}


def _is_muon_adamw_param(param_name: str, param: nn.Parameter) -> bool:
    """Returns True if this param should use AdamW when the main optimizer is Muon."""
    if param.ndim == 1:
        return True
    return any(name in param_name for name in _MUON_ADAMW_PARAM_NAMES)


class _MuonWithAdamW:
    """Wraps a Muon optimizer and an AdamW optimizer into a single optimizer-like object.

    Muon handles 2D+ weight matrices; AdamW handles embeddings, lm_head, and 1D params.
    """

    def __init__(self, muon: TorchMuon | None, adamw: TorchAdamW | None) -> None:
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self) -> list[dict]:
        groups = []
        if self.muon is not None:
            groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        return groups

    def step(self) -> None:
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

    def zero_grad(self) -> None:
        if self.muon is not None:
            self.muon.zero_grad()
        if self.adamw is not None:
            self.adamw.zero_grad()

    def state_dict(self) -> dict:
        return {
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if self.muon is not None and state_dict["muon"] is not None:
            self.muon.load_state_dict(state_dict["muon"])
        if self.adamw is not None and state_dict["adamw"] is not None:
            self.adamw.load_state_dict(state_dict["adamw"])

    def __repr__(self) -> str:
        return f"MuonWithAdamW(\n  muon={self.muon},\n  adamw={self.adamw}\n)"


def get_optimizer_container(
    optimizer_class_name: str,
    optimizer_class_args: dict,
    model_container: ModelContainer,
    params_group_method: ParamsGroupMethod,
    use_optimizer_with_backward_hook: bool,
) -> OptimizerContainer:
    """setup list of optimizers for the model

    Args:
        optimizer_class_name (str): optimizer class name
        optimizer_class_args (dict): args for the optimizer class
        model_container (ModelContainer): model container
        params_group_method (ParamsGroupMethod): the params grouping to use
        use_optimizer_with_backward_hook (bool): whether to use optimizer as a backward hook

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
        for model, params_groups in zip(model_container, params_groups_list):
            for param_name, param in model.named_parameters():
                if get_optimizer_split_function(param) is not None:
                    assert optimizer_class not in _SPLIT_FUNCTION_INCOMPATIBLE_OPTIMIZERS

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
    elif optimizer_class_name == "TorchMuon":
        adamw_args = {"lr": optimizer_class_args.get("lr", 1e-3)}
        optimizer_list_entries = []
        for params_groups in params_groups_list:
            muon_groups = []
            adamw_groups = []
            for group in params_groups.params_groups:
                muon_params = []
                adamw_params = []
                for param_name, param in group.parameter_name_map.items():
                    if _is_muon_adamw_param(param_name, param):
                        adamw_params.append(param)
                    else:
                        split_fn = get_optimizer_split_function(param)
                        muon_params.extend(split_fn(param) if split_fn is not None else [param])
                if muon_params:
                    muon_groups.append({"params": muon_params, **group.params_group_kwargs})
                if adamw_params:
                    adamw_groups.append({"params": adamw_params, **group.params_group_kwargs})

            muon = TorchMuon(muon_groups, **optimizer_class_args) if muon_groups else None
            adamw = TorchAdamW(adamw_groups, **adamw_args) if adamw_groups else None

            optimizer_list_entries.append(_MuonWithAdamW(muon, adamw))

        optimizer_list = OptimizerContainer(optimizer_list_entries)
    else:
        optimizer_list_entries = []
        for params_groups in params_groups_list:
            torch_params_groups = params_groups.to_torch_compatible_params_groups()
            for group in torch_params_groups:
                split_params = []
                for param in group["params"]:
                    split_fn = get_optimizer_split_function(param)
                    split_params.extend(split_fn(param) if split_fn is not None else [param])

                group["params"] = split_params

            optimizer_list_entries.append(optimizer_class(torch_params_groups, **optimizer_class_args))

        optimizer_list = OptimizerContainer(optimizer_list_entries)

    return optimizer_list
