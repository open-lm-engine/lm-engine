# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations


class MetricsTrackingDict:
    def __init__(self, data: dict) -> None:
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
                self.data[key] /= x
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
