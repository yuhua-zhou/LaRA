import json
import torch
from model.performance_preditor import PerformancePredictor
from utils.utils import read_layer_info

action_space = [2, 4, 6, 8, 10, 12, 14, 16]


def get_model_setting(file_path):
    model_settings = json.load(open(file_path))
    seen = []

    result = []
    for setting in model_settings:
        name = setting["name"].replace("-", "").replace("_", "-")
        if name not in seen:
            result.append({
                "name": name,
                "prune_list": setting["pruning_rate_list"],
                "layer_info": read_layer_info(name)
            })

            seen.append(name)

    return result


# 获取编码器
predictor = PerformancePredictor().double()
predictor.load_state_dict(torch.load("./output/performance_weights.pth"))
model_settings = get_model_setting("dataset/merged_file_v1.json")

for model in model_settings:
    name = model["name"]
    layer_info = torch.tensor(model["layer_info"], dtype=torch.float64)
    prune_list = torch.tensor(model["prune_list"], dtype=torch.float64)
    prune_list = prune_list.unsqueeze(1)

    best_performance = 0
    best_ranks = []


    def dfs_ranks(pos, rank_list, layer_info, prune_list):
        if pos == 32:
            ranks = torch.tensor(rank_list, dtype=torch.float64).unsqueeze(1)
            rank_list = ranks.unsqueeze(0)
            layer_info = layer_info.unsqueeze(0)
            prune_list = prune_list.unsqueeze(0)

            output = predictor(layer_info, rank_list, prune_list)
            performance = torch.sum(output)

            global best_ranks
            global best_performance

            if performance > best_performance:
                best_ranks = rank_list
            return

        for rank in action_space:
            dfs_ranks(pos + 1, rank_list + [rank], layer_info, prune_list)


    dfs_ranks(0, [], layer_info, prune_list)
    print(name, best_performance, "\n", best_ranks)
