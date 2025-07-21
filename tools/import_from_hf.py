# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from lm_engine.hf_models import import_from_huggingface


load_path = "load/"
save_path = "save/"

# export to HF llama
import_from_huggingface(load_path, save_path)
