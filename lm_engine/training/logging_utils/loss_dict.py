# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from torch.utils._pytree import register_pytree_node


class MetricsTrackingDict:
    def __init__(self, data: dict) -> MetricsTrackingDict:
        self.data = data

    def __add__(self, x: MetricsTrackingDict | dict | float | int) -> MetricsTrackingDict:
        if isinstance(x, (MetricsTrackingDict, dict)):
            if isinstance(x, MetricsTrackingDict):
                x = x.data

            for key, value in x.items():
                self.data[key] = self.data.get(key, 0) + value
        elif isinstance(x, (int, float)):
            for key in self.data:
                self.data[key] += x
        else:
            raise ValueError()

        return self

    def __truediv__(self, x: MetricsTrackingDict | dict | float | int) -> MetricsTrackingDict:
        if isinstance(x, (MetricsTrackingDict, dict)):
            if isinstance(x, MetricsTrackingDict):
                x = x.data

            for key, value in x.items():
                self.data[key] = self.data.get(key, 0) / value
        elif isinstance(x, (int, float)):
            for key in self.data:
                self.data[key] = self.data[key] / x
        else:
            raise ValueError()

        return self

    def get_dict(self) -> dict:
        return self.data

    def __iter__(self):
        for key in self.data:
            yield key

    def __getitem__(self, key: str) -> float:
        return self.data[key]

    def __setitem__(self, key: str, value: float) -> None:
        self.data[key] = value

    def __repr__(self) -> str:
        x = ""
        for key in self.data:
            x += f"{key} = {self[key]}\n"
        return x.rstrip()


# FSDP2 relies on pytree to find the tensors in a module's forward output so it can attach the
# hook that re-gathers root-level parameters before backward. Without this registration, tensors
# nested inside a MetricsTrackingDict (e.g. the loss) are invisible to FSDP2, and root parameters
# that aren't part of a nested fully_shard'd block (e.g. tied embeddings, lm_head) stay resharded
# with zero-size storage when backward runs, raising a `setStorage ... storage size 0` error.
register_pytree_node(
    MetricsTrackingDict,
    lambda instance: (list(instance.data.values()), list(instance.data.keys())),
    lambda values, keys: MetricsTrackingDict(dict(zip(keys, values))),
)
