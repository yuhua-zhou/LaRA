# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import csv

# tokenizer = AutoTokenizer.from_pretrained("baffo32/decapoda-research-llama-7B-hf")
model = AutoModelForCausalLM.from_pretrained("baffo32/decapoda-research-llama-7B-hf", local_files_only=True)

llama_model = model.get_decoder()
layers = llama_model.layers

import time

start = time.time()
model_code = np.array([])
for i, layer in enumerate(layers):
    layer_code = np.array([])
    parameters = layer.state_dict()
    weight_names = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    for name in weight_names:
        weight = parameters[name]
        # 执行奇异值分解
        U, S, V = np.linalg.svd(weight, full_matrices=False)

        layer_code = np.append(layer_code, S)
        print("layer: ", i, "奇异值 S:", S.shape)

    model_code = np.append(model_code, layer_code)

np.save("llama.npy", model_code)
print(time.time() - start)
