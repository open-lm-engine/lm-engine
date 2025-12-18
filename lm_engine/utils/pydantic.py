# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from .logger import log_rank_0


class BaseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def to_dict(self) -> dict:
        copied = deepcopy(self)

        for key, value in copied:
            if isinstance(value, BaseArgs):
                result = value.to_dict()
            elif isinstance(value, list):
                result = []
                for v in value:
                    if isinstance(v, BaseArgs):
                        result.append(v.to_dict())
            elif isinstance(value, Enum):
                result = value.value
            elif isinstance(value, type):
                result = value.__name__
            else:
                result = value

            setattr(copied, key, result)

        return vars(copied)

    @contextmanager
    def temporary_argument_value(self, name: str, value: Any):
        original_value = getattr(self, name)
        setattr(self, name, value)

        yield

        setattr(self, name, original_value)

    def log_args(self) -> None:
        def _iterate_args_recursively(args: BaseArgs, prefix: str = "") -> None:
            result = []

            if isinstance(args, BaseArgs):
                args = vars(args)

            p = len(prefix)

            for k, v in args.items():
                suffix = "." * (48 - len(k) - p)

                if isinstance(v, (BaseArgs, dict)):
                    if isinstance(v, dict) and len(v) == 0:
                        result.append(f"{prefix}{k} {suffix} " + r"{}")
                    else:
                        kv_list_subargs = _iterate_args_recursively(v, prefix + " " * 4)
                        result.append(f"{prefix}{k}:\n" + "\n".join(kv_list_subargs))
                elif isinstance(v, list) and all([isinstance(v_, (BaseArgs, dict)) for v_ in v]):
                    kv_list_subargs = []
                    for v_ in v:
                        v_ = _iterate_args_recursively(v_, prefix + " " * 4)
                        kv_list_subargs.append(f"\n".join(v_))
                    result.append(
                        f"{prefix}{k}:\n" + ("\n" + " " * (p + 4) + "*" * (44 - p) + "\n").join(kv_list_subargs)
                    )
                else:
                    result.append(f"{prefix}{k} {suffix} " + str(v))

            result.sort(key=lambda x: x.lower())
            return result

        log_rank_0(logging.INFO, "------------------------ arguments ------------------------")
        for line in _iterate_args_recursively(self):
            line = line.split("\n")
            for l in line:
                log_rank_0(logging.INFO, l)
        log_rank_0(logging.INFO, "-------------------- end of arguments ---------------------")
