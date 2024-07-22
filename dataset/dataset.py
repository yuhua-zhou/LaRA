import numpy as np
import torch
from torch.utils.data import Dataset
import json
from utils.utils import load_layer_info


class PerformanceDataset(Dataset):
    def __init__(self, file_path, prune_path, mode="train"):
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

        self.model_map = load_layer_info(prune_path)

    def statistics(self):
        data = self.data

        metrics = ['arc_easy', 'arc_challenge', 'winogrande', 'openbookqa', 'boolq', 'piqa', 'hellaswag']

        result = {}

        for d in data:
            for metric in metrics:
                if not metric in result.keys():
                    result[metric] = []

                result[metric].append(d["performance"][metric])

        print(result)

        for key, value in result.items():
            v = np.array(value)
            print("%s, 最大值: %.6f, 最小值: %.6f, 平均值: %.6f, 中位数: %.6f, 方差: %.6f" %
                  (key, np.max(v), np.min(v), np.mean(v), np.median(v), np.std(v)))

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


if __name__ == "__main__":
    train_set = PerformanceDataset("./merged_file_v1.json", "../rankadaptor/prune_log/local/")
    train_set.statistics()

    # a = np.array([1, 2, 3])
    # b = np.array([4, 5, 6])
    #
    # print(a + b)
