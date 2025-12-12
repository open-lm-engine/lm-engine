# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from lm_engine.arguments import TrainingArgs

from .test_commons import TestCommons


class ArgumentsOverrideTest(TestCommons):
    def test_argument_overrides(self) -> None:
        config = TestCommons.load_training_args_for_unit_tests("arguments_override.yml")
        keys = self._get_terminal_keys(config.to_dict())

        for key in keys:
            updated_config = TestCommons.load_training_args_for_unit_tests("arguments_override.yml").to_dict()

            value = updated_config
            key_split = key.split(".")
            for key in key_split[:-1]:
                value = value[key]

            value[key] = 1

            updated_config = TrainingArgs(**updated_config)

        assert False

    def _get_terminal_keys(self, config: dict) -> list[str]:
        keys = []
        for key, value in config.items():
            if isinstance(value, dict):
                keys += [f"{key}.{i}" for i in self._get_terminal_keys(value)]
            else:
                keys += [key]

        return keys
