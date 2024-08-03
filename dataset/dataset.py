import copy
import json
import random

import numpy as np
import torch
from prettytable import PrettyTable
from torch.utils.data import Dataset

from utils.utils import load_layer_info, softmax


class PerformanceDataset(Dataset):
    def __init__(self, file_path, prune_path, mode="train", augment=-1):
        super(PerformanceDataset, self).__init__()
        json_file = json.load(open(file_path, "r+"))
        self.data = []

        for item in json_file:
            # 13B还不考虑
            if "13b" not in item["name"]:
                self.data.append(item)

        random.seed(20240816)
        random.shuffle(self.data)

        if augment > 0:
            self.data_augmentation(augment)

        n_split = int(9 * len(self.data) / 10)

        if mode == "train":
            self.data = self.data[:n_split]
        elif mode == "test":
            self.data = self.data[n_split:]

        self.model_map = load_layer_info(prune_path, "pca")

    def statistics(self):
        data = self.data

        # ------------------------- compute data count -------------------------
        result = {}

        table = PrettyTable()
        table.field_names = ["model", "count"]

        for d in data:
            name = d["name"]
            if name not in result.keys():
                result[name] = 0
            result[name] += 1

        mx_count = 0
        for key, value in result.items():
            table.add_row([key, value])
            mx_count = max(mx_count, value)

        print(table)

        # ------------------------- compute data statistics -------------------------
        table = PrettyTable()
        table.field_names = ["metric", "max", "min", "avg", "mid", "std"]
        table.align = 'l'

        metrics = ['arc_easy', 'arc_challenge', 'winogrande', 'openbookqa', 'boolq', 'piqa', 'hellaswag']
        result = {metric: [] for metric in metrics}

        for d in data:
            performance = d["performance"]
            for metric in metrics:
                result[metric].append(performance[metric])

        for key, value in result.items():
            v = np.array(value)
            table.add_row([key, np.max(v), np.min(v), np.mean(v), np.median(v), np.std(v)])
        print(table)

    def data_augmentation(self, target):
        data = self.data
        aug_data = []

        result = {}

        for d in data:
            name = d["name"]
            if name not in result.keys():
                result[name] = []
            result[name].append(d)

        prob = {key: 1 / len(value) for key, value in result.items()}
        values = softmax(list(prob.values()))
        prob = {key: values[i] for i, key in enumerate(prob.keys())}

        # random.seed(None)

        def _swap_ranks(d):
            [i, j] = random.sample(range(4, 30), 2)
            d[i], d[j] = d[j], d[i]

        def _modify_ranks(d):
            i = random.randint(4, 29)
            actions = [2, 4, 6, 8, 10, 12, 14, 16]
            actions.remove(d[i])
            d[i] = random.choice(actions)

        models = list(prob.keys())
        weights = list(prob.values())
        for _ in range(target):
            [model] = random.choices(models, weights=weights, k=1)
            item = random.choice(result[model])
            new_item = copy.deepcopy(item)

            if random.random() > 0.5:
                _swap_ranks(new_item["rank_list"])
            else:
                _modify_ranks(new_item["rank_list"])

            aug_data.append(new_item)

        self.data = self.data + aug_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        name = item["name"]
        name = name.replace("-", "").replace("_", "-")

        # layer_info
        layer_info = self.model_map[name]
        layer_info = torch.from_numpy(layer_info).to(torch.float64)

        # rank_list
        rank_list = item["rank_list"]
        rank_list = torch.from_numpy(np.array(rank_list))
        rank_list = torch.unsqueeze(rank_list, 1).to(torch.float64)

        # prune_list
        prune_list = item["pruning_rate_list"]
        prune_list = torch.from_numpy(np.array(prune_list))
        prune_list = torch.unsqueeze(prune_list, 1).to(torch.float64)

        # performance
        performance = item["performance"]
        keys = list(performance.keys())
        keys.sort()
        performance = torch.tensor([performance[key] for key in keys], dtype=torch.float64)

        return (layer_info, rank_list, prune_list, performance)


if __name__ == "__main__":
    file_path = "merged_file_v2.json"
    prune_path = "../rankadaptor/prune_log/local/"
    mode = "train"
    target = 100
    train_set = PerformanceDataset(file_path, prune_path, mode, target)
    train_set.statistics()

    # layer_info, _, _, _ = train_set[0]
    # print(layer_info.shape)