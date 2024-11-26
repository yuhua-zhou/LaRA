import json
import random

import math
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model.nas.nas_model import PolicyNetwork, NASAgent, NASEnvironment, ReplayBuffer
from model.performance.performance_preditor import PerformancePredictor
from utils.plot import draw_loss_plot
from utils.utils import read_layer_info_svd


# get model setting
def get_model_setting(file_path, mode="local"):
    model_settings = json.load(open(file_path))
    model_settings = model_settings[mode]

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
def build_batch(data_list, batch_size):
    names = []
    prune_lists = []
    layer_infos = []

    sampled_models = random.sample(data_list, batch_size)

    for model in sampled_models:
        name = model["name"]

        layer_info = torch.tensor(model["layer_info"], dtype=torch.float64, device=device)
        prune_list = torch.tensor(model["pruning_rate_list"], dtype=torch.float64, device=device)
        prune_list = prune_list.unsqueeze(1)

        names.append(name)
        prune_lists.append(prune_list)
        layer_infos.append(layer_info)

    return (names,
            torch.stack(layer_infos),
            torch.stack(prune_lists))


# explore environment
def explore_environment(data_list, env, agent, replay_buffer):
    agent.set_temperature(2)

    print("exploring environment... ...")

    max_reward = 0.0
    min_reward = 100.0

    n = replay_buffer.buffer_size // model_batch_size + 1
    for _ in tqdm(range(n)):
        batch = build_batch(data_list, model_batch_size)

        # (batch, 1), (batch, 32, 6, 4096), (batch, 32, 1)
        (name, layer_infos, prune_lists) = batch

        state = (layer_infos, prune_lists)
        probs, log_probs, actions = agent.select_action(state)
        [performance, rewards] = env.step(actions, layer_infos, prune_lists)

        batch = list(zip(name, layer_infos, prune_lists, actions, rewards))
        replay_buffer.push(batch)

        max_reward = max(max_reward, sum(rewards))
        min_reward = min(min_reward, sum(rewards))

    print(f"explor end, max reward={max_reward / model_batch_size}, min reward={min_reward / model_batch_size}")


# train policy model
def train_policy_model(data_list, env, agent, policy_net, replay_buffer):
    agent.set_temperature(1)

    train_loss = []
    train_rewards = []
    train_entropy = []

    for epoch in range(num_epoch):
        new_temperature = agent.temperature / (1 + math.log(epoch // 1000 + 1))
        print(f"learning rate = {scheduler.get_last_lr()}, temperature = {new_temperature}")

        # if epoch // 10000 == 0:
        #     agent.set_temperature(agent.temperature / 2)

        # get from buffer
        batch = replay_buffer.sample(buffer_batch_size)

        # (batch, 32, 6, 4096), (batch, 32, 1), (batch, 32), (batch, 1)
        (layer_infos, prune_lists, action_lists, rewards) = batch
        action_lists = action_lists.unsqueeze(2)  # (batch, 32, 1)

        logits = policy_net(layer_infos, prune_lists)
        probs = F.softmax(logits / agent.temperature, dim=2)  # (batch, 32, 8)
        probs = torch.gather(probs, dim=2, index=action_lists)
        probs = probs.squeeze(2)
        log_probs = torch.log(probs)

        loss = agent.update_policy(log_probs, rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(action_lists)
        # print("grad of preditor")
        # for name, param in predictor.named_parameters():
        #     if param.grad:
        #         print(name, param.grad)
        #
        # print("grad of policy")
        # for name, param in policy_net.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad)

        # test and add test items
        with torch.no_grad():
            batch = build_batch(data_list, len(data_list))

            # (batch, 1), (batch, 32, 6, 4096), (batch, 32, 1)
            (name, layer_infos, prune_lists) = batch

            state = (layer_infos, prune_lists)
            probs, log_probs, actions = agent.select_action(state)

            # print(probs)
            print(actions)

            entropy = -(probs * torch.log(probs)).sum()  # 策略熵
            [performance, reward] = env.step(actions, layer_infos, prune_lists)

            new_batch = list(zip(name, layer_infos, prune_lists, actions, reward))
            replay_buffer.push(new_batch)

            loss = agent.update_policy(log_probs, reward)

        running_loss = loss.item()
        running_reward = torch.sum(reward).detach().item()
        running_entropy = entropy.detach().item()

        print("epoch %d: reward: %s, loss: %s, entropy: %s\n"
              % (epoch, running_reward, running_loss, running_entropy))

        train_loss.append(running_loss)
        train_rewards.append(running_reward)
        train_entropy.append(running_entropy)

        scheduler.step()
        # agent.set_temperature(new_temperature)

    draw_loss_plot(y=[train_loss], title="training loss", labels=["training loss"])
    draw_loss_plot(y=[train_rewards], title="training reward", labels=["training reward"])
    draw_loss_plot(y=[train_entropy], title="training entropy", labels=["training entropy"])

    import json

    result = {
        "training_loss": [train_loss],
        "training_rewards": [train_rewards],
        "training_entropy": [train_entropy]
    }

    with open("nas.json", "w+") as file:
        result = json.dumps(result)
        file.write(result)

    # torch.save(policy_net.state_dict(), "output/nas_weights_v1.pth")


if __name__ == "__main__":
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    action_space = [2, 4, 6, 8, 10, 12, 14, 16]
    embed_size = 128
    num_epoch = 30000
    model_batch_size = 4
    buffer_size = 500
    buffer_batch_size = 64
    lr = 1e-3

    # 模型设置
    model_settings = get_model_setting("dataset/model_settings.json")

    # 加载性能模型
    predictor = PerformancePredictor(input_size=128, hidden_size=128).double().to(device)
    predictor.load_state_dict(
        torch.load("./output/performance_weights_svd_V100_new.pth", weights_only=False,
                   map_location=torch.device(device)))
    predictor.eval()

    # 加载策略网络模型
    policy_net = PolicyNetwork(embed_size, 128, len(action_space)).double().to(device)
    policy_net.load_state_dict(
        torch.load("./output/nas_weights_pretrain.pth", weights_only=False, map_location=torch.device(device)))
    policy_net.train()

    # 加载环境和代理
    env = NASEnvironment(action_space, predictor, device=device)
    agent = NASAgent(policy_net, env)

    # 定义优化器
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-5)

    replay_buffer = ReplayBuffer(buffer_size)

    # explore
    explore_environment(model_settings, env, agent, replay_buffer)
    # train
    train_policy_model(model_settings, env, agent, policy_net, replay_buffer)
