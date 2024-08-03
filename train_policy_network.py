import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.nas_model2 import PolicyNetwork
from utils.utils import read_layer_info_svd
from utils.plot import draw_loss_plot

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available(), device)

action_space = [2, 4, 6, 8, 10, 12, 14, 16]
embed_size = 64
num_epoch = 10
batch_size = 2
lr = 1e-3

random.seed(20240816)


def get_model_setting(file_path):
    model_settings = json.load(open(file_path))
    seen = []

    base_path = "./rankadaptor/prune_log/local/"

    result = []
    for setting in model_settings:
        name = setting["name"].replace("-", "").replace("_", "-")
        if name not in seen:
            result.append({
                "name": name,
                "prune_list": setting["pruning_rate_list"],
                "layer_info": read_layer_info_svd(base_path + name)
            })

            seen.append(name)

    return result


def build_batch(data_list, batch_size):
    names = []
    prune_lists = []
    layer_infos = []
    budget_lists = []

    for i in range(batch_size):
        model = random.choice(data_list)
        name = model["name"]

        layer_info = torch.tensor(model["layer_info"], dtype=torch.float64, device=device)
        prune_list = torch.tensor(model["prune_list"], dtype=torch.float64, device=device)
        prune_list = prune_list.unsqueeze(1)

        names.append(name)
        prune_lists.append(prune_list)
        layer_infos.append(layer_info)
        budget_lists.append(torch.tensor([256], dtype=torch.float64, device=device))

    return (names,
            torch.stack(layer_infos),
            torch.stack(prune_lists),
            torch.stack(budget_lists)
            )


net = PolicyNetwork(embed_size, 128, len(action_space)).double().to(device)
# 模型设置
model_settings = get_model_setting("dataset/merged_file_v1.json")
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

loss_list = []
for epoch in range(num_epoch):
    names, layer_infos, prune_lists, budget_list = build_batch(data_list=model_settings, batch_size=batch_size)

    # batch_size, seq_length, _ = prune_lists.shape
    # print(budget_list)
    # budget_list = budget_list / (seq_length * (action_space[-1] - action_space[0]))
    # print(budget_list)

    # embedding as default state
    logits = net(layer_infos, prune_lists, budget_list)
    probs = F.softmax(logits, dim=2)
    print(probs.shape)

    predicted_budget = probs * action_space
    print(predicted_budget.shape)

    # action_indices = torch.argmax(probs, dim=2)
    #
    # actions = torch.tensor(action_space)[action_indices]
    # actions = actions.to(torch.float64)
    #
    # predicted_budget = torch.sum(actions, dim=1).unsqueeze(1)

    loss = criterion(predicted_budget, budget_list)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"{names}, predicted={predicted_budget}, target={budget_list}, loss={loss.item()}")

draw_loss_plot(y=[loss_list], title="total loss", labels=["train_loss"])
torch.save(net.state_dict(), "./output/nas_weights.pth")
