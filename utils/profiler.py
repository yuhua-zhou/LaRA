# Load model directly
from transformers import AutoModelForCausalLM
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time


class Profiler():
    def __init__(self, huggingface_model, full_param=(4096, 4096)):
        decoder = huggingface_model.get_decoder()
        self.layers = decoder.layers
        self.full_param = full_param

    def profile_layer(self, layer, layer_position):

        layer_encoding = []
        parameters = layer.state_dict()

        weight_names = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight"
        ]

        for name in weight_names:
            weight = parameters[name]
            weight = weight.numpy().astype(np.float32)
            # 执行奇异值分解
            U, S, V = np.linalg.svd(weight, full_matrices=False)

            layer_encoding.append(S.tolist())
            print(layer_position, "finish, ", U.shape, S.shape, V.shape)

        return layer_encoding
        # return {
        #     "pruning_rate": 1,
        #     "layer_position": layer_position,
        # }

    def profile(self):
        start_time = time.time()

        model_encoding = []

        with ThreadPoolExecutor() as executor:
            layer_encodings = [executor.submit(self.profile_layer, layer, i) for i, layer in enumerate(self.layers)]
            for layer_encoding in layer_encodings:
                model_encoding.append(layer_encoding.result())

        print(np.array(model_encoding).shape)
        print("finishing model profiling : ", time.time() - start_time)
        return model_encoding


model = AutoModelForCausalLM.from_pretrained("baffo32/decapoda-research-llama-7B-hf", local_files_only=True)

profiler = Profiler(huggingface_model=model)
result = profiler.profile()
