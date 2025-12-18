# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import AutoModelForCausalLM, AutoTokenizer

import lm_engine.hf_models


model_path = "unsharded_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


x = "Mayank"
x = tokenizer(x, return_tensors="pt")

x = model.generate(**x)
print(tokenizer.batch_decode(x))
