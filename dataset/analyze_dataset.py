import json
from utils.utils import load_layer_info
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsn_plot_scatter(data):
    tsne = TSNE(n_components=2, random_state=20240816)
    data_embed = tsne.fit_transform(data)

    # 绘制2D散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_embed[:, 0], data_embed[:, 1], c=np.arange(data.shape[0]), cmap='viridis', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Combined Data')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.show()


def distribution(data, model_map):
    avg_pool = nn.AdaptiveAvgPool1d(128)
    for key, value in model_map.items():
        layer_info = torch.from_numpy(value)

        seq_len, weight_num, hidden = layer_info.shape
        layer_info = layer_info.view(seq_len, weight_num * hidden)
        layer_info = avg_pool(layer_info)
        layer_info = layer_info.view(-1)
        layer_info = layer_info.detach().numpy()

        # print(key, len(layer_info), sum(layer_info))
        # print(layer_info)
        model_map[key] = layer_info

    samples = []
    for item in data:
        name = item["name"]
        name = name.replace("-", "").replace("_", "-")
        layer_info = model_map[name]

        rank_list = np.array(item["rank_list"])
        prune_list = np.array(item["pruning_rate_list"])

        performance = item["performance"]
        keys = list(performance.keys())
        keys.sort()
        performance = np.array([performance[key] for key in keys])

        samples.append(np.concatenate((layer_info, prune_list, rank_list, performance)))

    samples = np.array(samples)
    print(samples.shape)
    tsn_plot_scatter(samples)


def compute_mean_std(model_map):
    result = None
    output_size = 64
    avg_pool = nn.AdaptiveAvgPool1d(output_size)
    for key, value in model_map.items():
        layer_info = torch.from_numpy(value)
        layer_info = layer_info[:, :, :output_size].clone()

        print(layer_info.shape)
        # seq_len, weight_num, hidden = layer_info.shape
        seq_len = layer_info.shape[0]
        layer_info = layer_info.view(seq_len, -1)
        layer_info = avg_pool(layer_info)

        layer_info = layer_info.detach().numpy()
        if result is None:
            result = layer_info
        else:
            result = np.concatenate([result, layer_info], axis=1)

    print(
        f"shape of result={result.shape}, mean={result.mean()}, std={result.std()}, max={result.max()}, min={result.min()}")

    result = (result - result.min()) / (result.max() - result.min())
    print(
        f"shape of result={result.shape}, mean={result.mean()}, std={result.std()}, max={result.max()}, min={result.min()}")


if __name__ == "__main__":
    file_path = "./merged_file_v2.json"
    prune_path = "../rankadaptor/prune_log/local/"
    data = json.load(open(file_path, "r+"))
    model_map = load_layer_info(prune_path, type="pca")

    # distribution(data, model_map)
    compute_mean_std(model_map)
