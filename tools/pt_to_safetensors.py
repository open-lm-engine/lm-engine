# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_engine.hf_models import register_model_classes


# register new model classes since the source models that you wish to convert to
# safetensors may be of one of these model architectures.
register_model_classes()

# path to the checkpoint that is a full state dict dump and you wish to
# create safetensors for
checkpoint_to_be_converted = "checkpoint/"

# path where safetensors have to be saved
safetensors_destination_path = "checkpoint-st/"

# loading the model with full precision
# you can modify this behaviour by passing torch_dtype=<intended dtype>
model = AutoModelForCausalLM.from_pretrained(checkpoint_to_be_converted, torch_dtype=torch.bfloat16).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# save_pretrained() by default saves in safetensors format
# does not move the tokenizer data
model.save_pretrained(safetensors_destination_path)

# to move tokenizer
# simply load from the source and save it to the destination path
tokenizer = AutoTokenizer.from_pretrained(checkpoint_to_be_converted)
tokenizer.save_pretrained(safetensors_destination_path, legacy_format=False)
