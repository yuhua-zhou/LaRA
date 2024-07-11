import numpy as np
import torch
from torch.utils.data import Dataset
import json
from utils.utils import load_layer_info


class PerformanceDataset(Dataset):
    def __init__(self, file_path, mode="train"):
        super(PerformanceDataset, self).__init__()
        json_file = json.load(open(file_path, "r+"))
        self.data = []

        for item in json_file:
            # 13B还不考虑
            if "13b" not in item["name"]:
                self.data.append(item)

        n_split = int(4 * len(self.data) / 5)

        if mode == "train":
            self.data = self.data[:n_split]
        elif mode == "test":
            self.data = self.data[n_split:]

        model_list = [
            "llama7b-0.20", "llama7b-0.25", "llama7b-0.30", "llama7b-0.50",
            "vicuna7b-0.20", "vicuna7b-0.25", "vicuna7b-0.30", "vicuna7b-0.50",
        ]

        self.model_map = load_layer_info(model_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        name = item["name"]
        name = name.replace("-", "").replace("_", "-")

        # layer_info
        layer_info = self.model_map[name]
        layer_info = torch.from_numpy(layer_info)

        # rank_list
        rank_list = item["rank_list"]
        rank_list = torch.from_numpy(np.array(rank_list))
        rank_list = torch.unsqueeze(rank_list, 1)

        # prune_list
        prune_list = item["pruning_rate_list"]
        prune_list = torch.from_numpy(np.array(prune_list))
        prune_list = torch.unsqueeze(prune_list, 1)

        # performance
        performance = item["performance"]
        performance = torch.from_numpy(np.array(list(performance.values())))

        return (layer_info, rank_list, prune_list, performance)
