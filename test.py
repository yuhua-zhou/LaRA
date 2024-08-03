# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import numpy as np
# import csv
#
# # tokenizer = AutoTokenizer.from_pretrained("baffo32/decapoda-research-llama-7B-hf")
# model = AutoModelForCausalLM.from_pretrained("baffo32/decapoda-research-llama-7B-hf", local_files_only=True)
# 
# llama_model = model.get_decoder()
# layers = llama_model.layers
# 
# import time
# 
# start = time.time()
# model_code = np.array([])
# for i, layer in enumerate(layers):
#     layer_code = np.array([])
#     parameters = layer.state_dict()
#     weight_names = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
#     for name in weight_names:
#         weight = parameters[name]
#         # 执行奇异值分解
#         U, S, V = np.linalg.svd(weight, full_matrices=False)
# 
#         layer_code = np.append(layer_code, S)
#         print("layer: ", i, "奇异值 S:", S.shape)
# 
#     model_code = np.append(model_code, layer_code)
# 
# np.save("llama.npy", model_code)
# print(time.time() - start)


# import numpy as np
#
# a = np.random.rand(256, 8) @ np.random.rand(8, 256)
# b = np.random.rand(256, 8) @ np.random.rand(8, 256)
# c = np.random.rand(256, 8) @ np.random.rand(8, 256)
#
# print(np.linalg.matrix_rank(a))
# print(np.linalg.matrix_rank(b))
# print(np.linalg.matrix_rank(c))
# print(np.linalg.matrix_rank(a + b + c))


import json
import os

from tqdm import tqdm

base_path = "./rankadaptor/results/"
files = os.listdir(base_path)
results = []

for file in tqdm(files):
    json_file = json.load(open(base_path + file, "r+"))

    performance = {}
    for key in json_file["results"].keys():
        # performance[key] = round(json_file["results"][key]["acc"], 4)
        performance[key] = json_file["results"][key]["acc"]

    item = {
        "name": file[:file.index("0.50") + 4],
        "rank_list": json_file["config"]["rank_list"],
        "pruning_rate_list": [
            0,
            0,
            0,
            0,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0,
            0
        ],
        "performance": performance
    }

    results.append(item)

print(results)

with open("data_20240720.json", "w+") as file:
    file.write(json.dumps(results))
