import numpy as np
from torch.utils.data import Dataset
import json


class PerformanceDataset(Dataset):
    def __init__(self, file_path):
        super(PerformanceDataset, self).__init__()
        json_file = json.load(open(file_path, "r+"))
        self.data = []

        for item in json_file:
            if "13B" not in item["name"]:
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        name = item["name"]
        rank_list = item["rank_list"]
        pruning_rate_list = item["pruning_rate_list"]
        performance = item["performance"]

        # layer_info = torch.randn(batch_size, seq_length, 6, 4096).to(torch.float32)
        # rank_list = torch.tensor([ranks for i in range(batch_size)], dtype=torch.float32)
        # prune_list = torch.tensor([prunes for i in range(batch_size)], dtype=torch.float32)

        return ()


if __name__ == "__main__":
    train_set = PerformanceDataset("./merged_file_revise.json")
    print(len(train_set))
    print(train_set[0])
