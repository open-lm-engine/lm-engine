# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os

from lm_engine.arguments import TrainingArgs
from lm_engine.utils import load_yaml

from ..test_common import BaseTestCommons


class TestCommons(BaseTestCommons):
    @staticmethod
    def load_training_args_for_unit_tests(filename: str) -> TrainingArgs:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        return TrainingArgs(**load_yaml(filepath))
