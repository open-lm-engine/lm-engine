# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from copy import deepcopy
from enum import Enum

from pydantic import BaseModel, ConfigDict


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
