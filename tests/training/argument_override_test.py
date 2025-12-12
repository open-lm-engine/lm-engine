# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import types
import typing

from lm_engine.arguments import TrainingArgs

from .test_commons import TestCommons


class ArgumentsOverrideTest(TestCommons):
    def test_argument_overrides(self) -> None:
        config = TestCommons.load_training_args_for_unit_tests("data_config.yml")
        keys = self._get_terminal_keys(config.to_dict())

        for key in keys:
            updated_config = TestCommons.load_training_args_for_unit_tests("data_config.yml").to_dict()

            value = updated_config
            key_split = key.split(".")
            for key in key_split[:-1]:
                value = value[key]

            # Get the field type from the pydantic model
            field_type = self._get_field_type(config, key_split)
            value[key_split[-1]] = self._get_default_value_for_type(field_type)

            updated_config = TrainingArgs(**updated_config)

        assert False

    def _is_union_type(self, field_type) -> bool:
        """Check if a type is a Union type (either typing.Union or types.UnionType for | syntax)."""
        origin = typing.get_origin(field_type)
        # Check for typing.Union (Optional[str], Union[str, None])
        if origin is typing.Union:
            return True
        # Check for types.UnionType (str | None syntax in Python 3.10+)
        if hasattr(types, "UnionType") and origin is types.UnionType:
            return True
        return False

    def _get_field_type(self, model: TrainingArgs, key_path: list[str]) -> type:
        """Get the type of a field from a pydantic model using a key path."""
        current_model = model.__class__

        for key in key_path:
            field_info = current_model.model_fields.get(key)
            if field_info is None:
                raise ValueError(f"Field {key} not found in {current_model}")

            # Get the annotation type
            field_type = field_info.annotation

            # If this is not the last key, navigate to the nested model
            if key != key_path[-1]:
                # Handle Optional types and unions
                if self._is_union_type(field_type):
                    # Get the non-None type from Optional/Union
                    args = typing.get_args(field_type)
                    field_type = next((arg for arg in args if arg is not type(None)), args[0])
                current_model = field_type

        # Unwrap Optional/Union types for the final field as well
        if self._is_union_type(field_type):
            args = typing.get_args(field_type)
            field_type = next((arg for arg in args if arg is not type(None)), args[0])

        return field_type

    def _get_default_value_for_type(self, field_type) -> any:
        """Generate an appropriate default value for a given type."""

        # Handle Union/Optional types
        if self._is_union_type(field_type):
            args = typing.get_args(field_type)
            # Get the first non-None type
            field_type = next((arg for arg in args if arg is not type(None)), args[0])

        # Basic type mappings
        if field_type == int or field_type == "int":
            return 1
        elif field_type == float or field_type == "float":
            return 1.0
        elif field_type == str or field_type == "str":
            return "test"
        elif field_type == bool or field_type == "bool":
            return True
        elif typing.get_origin(field_type) == list:
            return []
        elif typing.get_origin(field_type) == dict:
            return {}
        else:
            # For enum or custom types, return 1 as fallback
            return 1

    def _get_terminal_keys(self, config: dict) -> list[str]:
        keys = []
        for key, value in config.items():
            if isinstance(value, dict):
                keys += [f"{key}.{i}" for i in self._get_terminal_keys(value)]
            else:
                keys += [key]

        return keys
