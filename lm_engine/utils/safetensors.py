# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import json
import os

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME


class SafeTensorsWeightsManager:
    def __init__(self, model_path: str) -> None:
        if model_path.endswith(".safetensors"):
            filenames = [model_path]
        else:
            filenames = os.listdir(model_path)
            filenames = filter(lambda f: f.endswith(".safetensors"), filenames)
            filenames = [os.path.join(model_path, filename) for filename in filenames]

        self.tensor_filenames = {}
        self.file_handles = {}

        for filename in filenames:
            f = safe_open(filename, framework="pytorch")
            self.file_handles[filename] = f

            for tensor_name in f.keys():
                self.tensor_filenames[tensor_name] = filename

    def get_slice(self, tensor_name: str):
        filename = self.tensor_filenames[tensor_name]
        f = self.file_handles[filename]
        return f.get_slice(tensor_name)

    def get_tensor(
        self, tensor_name: str, dtype: torch.dtype | None = None, device: torch.device | None = None
    ) -> torch.Tensor:
        filename = self.tensor_filenames[tensor_name]
        f = self.file_handles[filename]
        tensor = f.get_tensor(tensor_name)
        tensor = tensor.to(dtype=dtype, device=device)
        return tensor

    def get_shape(self, tensor_name: str) -> torch.Tensor:
        slice = self.get_slice(tensor_name)
        return slice.get_shape()

    def has_tensor(self, tensor_name: str) -> bool:
        return tensor_name in self.tensor_filenames

    def __len__(self) -> int:
        return len(self.tensor_filenames)

    def __iter__(self):
        for tensor_name in self.tensor_filenames:
            yield tensor_name

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, SafeTensorsWeightsManager):
            return False

        if len(self) != len(__value):
            return False

        for tn1, tn2 in zip(self, __value):
            if tn1 != tn2:
                return False

            if not self.get_tensor(tn1).equal(__value.get_tensor(tn2)):
                return False

        return True

    def state_dict(self) -> dict:
        return {tensor_name: self.get_tensor(tensor_name) for tensor_name in self}

    @staticmethod
    def save_state_dict(state_dict: dict, save_path: str) -> None:
        os.makedirs(save_path, exist_ok=True)

        state_dict_split = split_torch_state_dict_into_shards(state_dict)
        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor] for tensor in tensors}
            save_file(
                shard,
                os.path.join(save_path, filename),
                metadata={"format": "pt"},
            )

        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }

            with open(os.path.join(save_path, SAFE_WEIGHTS_INDEX_NAME), "w") as f:
                f.write(json.dumps(index, indent=2))
