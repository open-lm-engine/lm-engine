# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from parameterized import parameterized

from lm_engine.arguments import TrainingArgs
from lm_engine.enums import DatasetSplit

from .test_common import BaseTestCommons


class ArgumentsOverrideTest(BaseTestCommons):
    @parameterized.expand([DatasetSplit.train, DatasetSplit.val, DatasetSplit.test])
    def test_argument_overrides(self, split: DatasetSplit) -> None:
        config = TrainingArgs()
        keys = self._recursively_get_keys(config.to_dict())
        print(keys)
        assert False

    def _recursively_get_keys(self, config: dict) -> list[str]:
        keys = []
        for key, value in config.items():
            keys.append(key)

            if isinstance(value, dict):
                _keys = self._recursively_get_keys(value)
                _keys = [f"{key}.{i}" for i in _keys]
                keys += _keys

        return keys
