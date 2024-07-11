import json
import random
import matplotlib.pyplot as plt
import torch
from model.performance_preditor import PerformancePredictor, PositionalEncoder
from model.nas_model import PolicyNetwork, NASAgent, NASEnvironment
from utils.utils import read_layer_info
from torch.autograd import Variable

device = "cuda:0" if torch.cuda.is_available() else "cpu"
action_space = [2, 4, 6, 8, 10, 12, 14, 16]
embed_size = 64
num_epoch = 100
num_episode = 100
batch_size = 4

# 获取编码器
predictor = PerformancePredictor().double().to(device)
predictor.load_state_dict(torch.load("./output/performance_weights.pth"))

pos_encoder = PositionalEncoder(num_hiddens=embed_size).to(device)

policy_net = PolicyNetwork(embed_size, 128, len(action_space)).double().to(device)
env = NASEnvironment(action_space, predictor, device=device)
agent = NASAgent(policy_net, env)


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


def build_batch(data_list, batch_size):
    names = []
    prune_lists = []
    layer_infos = []

    for i in range(batch_size):
        model = random.choice(data_list)
        name = model["name"]

        layer_info = torch.tensor(model["layer_info"], dtype=torch.float64, device=device)
        prune_list = torch.tensor(model["prune_list"], dtype=torch.float64, device=device)
        prune_list = prune_list.unsqueeze(1)

        names.append(name)
        prune_lists.append(prune_list)
        layer_infos.append(layer_info)

    return (names,
            torch.stack(layer_infos),
            torch.stack(prune_lists))


# draw loss plot
def draw_loss_plot(x, y, name="training"):
    plt.plot(x, y, label=name + " loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


# 模型设置
model_settings = get_model_setting("./dataset/merged_file_revise.json")

train_loss = []
train_rewards = []

for epoch in range(num_epoch):
    for episode in range(num_episode):
        names, layer_infors, prune_lists = build_batch(data_list=model_settings, batch_size=batch_size)

        # embedding as default state
        state = (layer_infors, prune_lists)

        # logits, actions = agent.select_action(state)
        probs, log_probs, actions = agent.select_action(state)
        reward = env.step(actions, layer_infors, prune_lists)

        # print(probs)

        # log_logits = logits.gather(dim=1, index=actions).squeeze(1)
        loss = agent.update_policy(log_probs, reward)

        print("models %s: epoch: %d, episode: %d, reward: %s, loss: %s \n\n"
              % (names, epoch, episode, reward.detach().item(), loss.detach().item()))

        train_loss.append(loss.detach().item())
        train_rewards.append(reward.detach().item())

draw_loss_plot(range(num_epoch * num_episode), train_loss, "loss")
draw_loss_plot(range(num_epoch * num_episode), train_rewards, "reward")
