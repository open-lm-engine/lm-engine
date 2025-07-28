# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from lm_engine.hf_models import export_to_huggingface


load_path = "load/"
save_path = "save/"

# export to HF llama
export_to_huggingface(load_path, save_path, model_type="llama")
