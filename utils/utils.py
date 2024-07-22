import numpy as np
import os


def read_layer_info(path):
    "./rankadaptor/prune_log/local/"
    layer_info = np.load(path + "/svd.npy", allow_pickle=True)
    layer_info = layer_info.tolist()
    new_info = []

    for layer in layer_info:
        new_layer = []
        for key in layer:
            arr = np.array(key)
            arr = np.pad(arr, (0, 4096 - arr.shape[0]))
            new_layer.append(arr)
        new_info.append(new_layer)

    return np.array(new_info)


def load_layer_info(path):
    model_list = os.listdir(path)

    model_map = dict()

    for model_name in model_list:
        model_map[model_name] = read_layer_info(path + model_name)

    return model_map
