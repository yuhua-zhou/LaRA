import numpy as np
import os

import torch


def read_layer_info_svd(path):
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


def read_layer_info_pca(path):
    "./rankadaptor/prune_log/local/"
    layer_info = np.load(path + "/pca.npy")
    return layer_info


def load_layer_info(path, type="svd"):
    model_list = os.listdir(path)

    model_map = dict()

    for model_name in model_list:
        if type == "svd":
            model_map[model_name] = read_layer_info_svd(path + model_name)
        elif type == "pca":
            model_map[model_name] = read_layer_info_pca(path + model_name)

    return model_map


def normalize(data, max=1, min=0, mean=0, std=1):
    # data = (data - min) / (max - min)
    data = (data - mean) / std
    return data


if __name__ == "__main__":
    model_map = load_layer_info("../rankadaptor/prune_log/local/", "pca")
    for key, value in model_map.items():
        print(key, value.shape)
