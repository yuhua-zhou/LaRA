import json
import random

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.nas.nas_model2 import PolicyNetwork, NASAgent, NASEnvironment
from model.performance.performance_preditor import PerformancePredictor
from utils.plot import draw_loss_plot
from utils.utils import read_layer_info_svd

device = "cuda:0" if torch.cuda.is_available() else "cpu"
action_space = [2, 4, 6, 8, 10, 12, 14, 16]
embed_size = 128
num_epoch = 100
n = 20
batch_size = 4
lr = 1e-3
random.seed(20240816)

# 获取编码器
predictor = PerformancePredictor(input_size=128, hidden_size=128).double().to(device)
predictor.load_state_dict(torch.load("./output/performance_weights_svd_4090.pth"))

policy_net = PolicyNetwork(embed_size, 128, len(action_space)).double().to(device)
policy_net.load_state_dict(torch.load("./output/nas_weights_pretrain.pth"))

env = NASEnvironment(action_space, predictor, device=device)
agent = NASAgent(policy_net, env, temperature=2)

# 定义优化器
optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-4)


# get model setting
def get_model_setting(file_path):
    model_settings = json.load(open(file_path))

    base_path = "./rankadaptor/prune_log/local/"

    result = [
        {
            "name": setting["name"],
            "pruning_rate_list": setting["pruning_rate_list"],
            "layer_info": read_layer_info_svd(base_path + setting["name"])
        } for setting in model_settings
    ]

    return result


# create dataset
def create_dataset(data_list, n, batch_size):
    dataset = []

    def _build_batch(data_list, batch_size):
        names = []
        prune_lists = []
        layer_infos = []
        budget_lists = []

        for i in range(batch_size):
            model = random.choice(data_list)
            name = model["name"]

            layer_info = torch.tensor(model["layer_info"], dtype=torch.float64, device=device)
            prune_list = torch.tensor(model["pruning_rate_list"], dtype=torch.float64, device=device)
            prune_list = prune_list.unsqueeze(1)

            names.append(name)
            prune_lists.append(prune_list)
            layer_infos.append(layer_info)

            if random.random() > 0.5:
                budget_lists.append(torch.tensor([256], dtype=torch.float64, device=device))
            else:
                budget_lists.append(torch.tensor([320], dtype=torch.float64, device=device))

        return (names,
                torch.stack(layer_infos),
                torch.stack(prune_lists),
                torch.stack(budget_lists))

    for i in range(n):
        dataset.append(_build_batch(data_list, batch_size))

    return dataset


# 模型设置
model_settings = get_model_setting("dataset/model_settings.json")
dataset = create_dataset(model_settings, n, batch_size)

train_loss = []
train_rewards = []

for epoch in range(num_epoch):
    policy_net.train()
    print(f"learning rate = {scheduler.get_last_lr()}")

    running_loss = 0.0
    running_reward = 0.0
    for batch in dataset:
        names, layer_infos, prune_lists, budget_list = batch

        # embedding as default state
        state = (layer_infos, prune_lists)

        # logits, actions = agent.select_action(state)
        probs, log_probs, actions = agent.select_action(state, budget_list)
        reward = env.step(actions, layer_infos, prune_lists, budget_list)

        loss = agent.update_policy(log_probs, reward)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_reward += torch.sum(reward).detach().item()

        print(actions)
        print("models %s: epoch: %d, reward: %s, loss: %s \n\n"
              % (names, epoch, torch.sum(reward).detach().item(), loss.detach().item()))

    train_loss.append(running_loss)
    train_rewards.append(running_reward)
    scheduler.step()
    agent.update_temperature(epoch)

draw_loss_plot(y=[train_loss], title="training loss", labels=["total loss"])
draw_loss_plot(y=[train_rewards], title="training reward", labels=["total reward"])

torch.save(policy_net.state_dict(), "output/nas_weights_v1.pth")
