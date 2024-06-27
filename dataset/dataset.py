import numpy as np
import torch
from torch.utils.data import Dataset
import json


class PerformanceDataset(Dataset):
    def __init__(self, file_path):
        super(PerformanceDataset, self).__init__()
        json_file = json.load(open(file_path, "r+"))
        self.data = []
        for item in json_file:
            # 13B还不考虑
            if "13b" not in item["name"]:
                self.data.append(item)

        self.model_map = self.load_layer_info()

    def load_layer_info(self):
        model_list = [
            "llama7b-0.20", "llama7b-0.25", "llama7b-0.30", "llama7b-0.50",
            "vicuna7b-0.20", "vicuna7b-0.25", "vicuna7b-0.30", "vicuna7b-0.50",
        ]

        def read_layer_info(path):
            layer_info = np.load("./rankadaptor/prune_log/local/" + path + "/svd.npy", allow_pickle=True)
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

        model_map = dict()

        for model_name in model_list:
            model_map[model_name] = read_layer_info(model_name)

        return model_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        name = item["name"]
        name = name.replace("-", "").replace("_", "-")
        layer_info = self.model_map[name]
        layer_info = torch.from_numpy(layer_info)

        rank_list = item["rank_list"]
        rank_list = torch.from_numpy(np.array(rank_list))
        rank_list = torch.unsqueeze(rank_list, 1)

        prune_list = item["pruning_rate_list"]
        prune_list = torch.from_numpy(np.array(prune_list))
        prune_list = torch.unsqueeze(prune_list, 1)

        performance = item["performance"]
        performance = torch.from_numpy(np.array(list(performance.values())))

        # layer_info = torch.randn(batch_size, seq_length, 6, 4096).to(torch.float32)
        # rank_list = torch.tensor([ranks for i in range(batch_size)], dtype=torch.float32)
        # prune_list = torch.tensor([prunes for i in range(batch_size)], dtype=torch.float32)

        # return (layer_info, rank_list, prune_list, performance)
        return (layer_info, rank_list, prune_list, performance)
