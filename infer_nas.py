import json

import torch
from tqdm import tqdm

from model.nas.nas_model import PolicyNetwork, NASAgent, NASEnvironment
from model.performance.performance_preditor import PerformancePredictor
from utils.utils import read_layer_info_svd

# get model setting
def get_model_setting(file_path, mode="local"):
    model_settings = json.load(open(file_path))
    model_settings = model_settings[mode]

    base_path = "./rankadaptor/prune_log/" + mode + "/"

    result = [
        {
            "name": setting["name"],
            "pruning_rate_list": setting["pruning_rate_list"],
            "layer_info": read_layer_info_svd(base_path + setting["name"])
        } for setting in model_settings
    ]

    return result

# sampling rank configurations
def sample_rank_configurations(model, env, agent, steps):
    agent.set_temperature(1.5)

    # (batch, 1), (batch, 32, 6, 4096), (batch, 32, 1)
    (name, layer_infos, prune_lists) = model

    print(name, layer_infos.shape, prune_lists.shape)
    print("sampling rank configurations... ...")

    budget = torch.tensor([[[256.0]]], device=device)

    max_reward = 0.0
    best_actions = torch.tensor([])

    best_configuration = ()

    for _ in tqdm(range(steps)):
        # for _ in range(steps):
        state = (layer_infos, prune_lists)
        probs, log_probs, actions = agent.select_action(state)
        # rewards = env.step(actions, layer_infos, prune_lists, budget_list=budget)
        [budget_list, rank_sum, penalty, performance, rewards] = env.step(actions, layer_infos, prune_lists,
                                                                          budget_list=budget)

        if sum(rewards) > max_reward:
            best_actions = actions.clone()
            best_configuration = (budget_list, rank_sum, penalty, performance, rewards)
            max_reward = sum(rewards)

    best_actions = best_actions * 2 + 2
    print(
        f"Explor end, max reward={max_reward}\n, best actions = {best_actions}\n, best configuration={best_configuration}")
    # print(f"Explor end, max reward={max_reward}, best actions = {best_actions}")
    print("--------------------------------------------------------------\n\n")

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    action_space = [2, 4, 6, 8, 10, 12, 14, 16]

    # 模型设置
    model_settings = get_model_setting("dataset/model_settings.json", mode="experiments")

    # 加载性能模型
    predictor = PerformancePredictor(input_size=128, hidden_size=128).double().to(device)
    predictor.load_state_dict(
        torch.load("./output/performance_weights_svd_V100_new.pth", map_location=torch.device(device),
                   weights_only=True))
    predictor.eval()

    # 加载策略网络模型
    policy_net = PolicyNetwork(128, 128, len(action_space)).double().to(device)
    policy_net.load_state_dict(
        torch.load("./output/nas_weights_v1.pth", map_location=torch.device(device), weights_only=True))
    policy_net.eval()

    # 加载环境和代理
    env = NASEnvironment(action_space, predictor, alpha=100, beta=1.2, device=device)
    agent = NASAgent(policy_net, env)

    for model in model_settings:
        name = model["name"]
        layer_info = torch.tensor(model["layer_info"], dtype=torch.float64, device=device)
        prune_list = torch.tensor(model["pruning_rate_list"], dtype=torch.float64, device=device)
        prune_list = prune_list.unsqueeze(1)

        layer_infos = torch.stack([layer_info])
        prune_lists = torch.stack([prune_list])

        # sample_rank_configurations((name, layer_infos, prune_lists), env, agent, 5000)

        if name in ["llama7b-0.35", "llama8b-0.50"]:
            sample_rank_configurations((name, layer_infos, prune_lists), env, agent, 10000)

# vicuna7b - 0.20 = 8, 8, 8, 8, 8, 8, 10, 2, 14, 2, 2, 8, 2, 2, 8, 2, 2, 12, 4, 4, 4, 8, 2, 10, 10, 8, 10, 10, 8, 8, 8, 8
# vicuna7b - 0.25 = 8, 8, 8, 8, 8, 8, 10, 10, 14, 2, 4, 10, 2, 8, 8, 8, 2, 2, 4, 2, 2, 4, 4, 6, 8, 10, 10, 10, 8, 8, 8, 8
# vicuna7b - 0.30 = 8, 8, 8, 8, 8, 10, 10, 2, 14, 2, 8, 2, 8, 12, 2, 2, 2, 2, 8, 2, 8, 8, 12, 8, 2, 8, 8, 10, 8, 8, 8, 8
# vicuna7b - 0.35 = 8, 8, 8, 8, 8, 8, 10, 2, 2, 4, 16, 8, 4, 12, 2, 2, 2, 2, 2, 12, 2, 6, 4, 4, 2, 4, 4, 8, 4, 8, 8, 8
# vicuna7b - 0.40 = 8, 8, 8, 8, 8, 8, 10, 2, 2, 2, 2, 8, 2, 2, 2, 2, 4, 16, 6, 2, 2, 4, 6, 4, 8, 2, 4, 8, 10, 8, 8, 8
# vicuna7b - 0.50 = 8, 8, 8, 8, 8, 4, 10, 2, 2, 16, 2, 2, 6, 12, 4, 4, 4, 4, 4, 2, 8, 4, 6, 6, 4, 4, 2, 8, 10, 8, 8, 8

# llama7b - 0.20 = 8, 8, 8, 8, 8, 8, 10, 2, 2, 2, 4, 2, 2, 2, 4, 2, 6, 2, 8, 4, 4, 8, 2, 12, 8, 2, 10, 10, 8, 8, 8, 8
# llama7b - 0.25 = 8, 8, 8, 8, 8, 8, 10, 2, 2, 12, 8, 2, 2, 8, 2, 8, 2, 4, 2, 2, 2, 8, 4, 10, 6, 8, 4, 4, 8, 8, 8, 8
# llama7b - 0.30 = 8, 8, 8, 8, 8, 4, 10, 2, 2, 2, 4, 8, 2, 2, 2, 2, 4, 16, 4, 2, 4, 2, 12, 4, 8, 4, 4, 10, 8, 10, 8, 8
# llama7b - 0.35 = 8, 8, 8, 8, 8, 8, 10, 2, 2, 16, 2, 4, 2, 8, 8, 6, 2, 2, 4, 4, 4, 8, 4, 6, 6, 8, 4, 10, 8, 8, 8, 8
# llama7b - 0.40 = 8, 8, 8, 8, 8, 2, 10, 2, 2, 16, 2, 2, 2, 8, 8, 6, 2, 2, 2, 2, 4, 12, 4, 6, 6, 6, 2, 4, 4, 14, 8, 8
# llama7b - 0.50 = 8, 8, 8, 8, 8, 8, 10, 2, 2, 2, 2, 2, 6, 6, 2, 2, 2, 4, 8, 2, 4, 16, 6, 10, 4, 8, 10, 2, 8, 8, 8, 8

# baichuan7b - 0.20 = 8, 8, 8, 8, 8, 8, 10, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 6, 8, 2, 4, 4, 4, 2, 10, 8, 10, 8, 14, 8, 8
# baichuan7b - 0.50 = 8, 8, 8, 8, 8, 4, 8, 14, 14, 16, 2, 2, 2, 2, 2, 2, 4, 4, 4, 2, 2, 4, 4, 8, 6, 4, 8, 16, 10, 8, 8, 8
# llama8b - 0.20 = 8, 8, 8, 8, 8, 10, 10, 2, 2, 4, 8, 2, 4, 2, 2, 2, 8, 4, 8, 8, 2, 8, 12, 8, 8, 10, 8, 8, 14, 8, 8, 8
# llama8b - 0.50 = 8, 8, 8, 8, 8, 4, 10, 2, 2, 2, 2, 8, 8, 12, 2, 2, 2, 2, 4, 16, 2, 4, 4, 2, 6, 4, 10, 10, 8, 8, 8, 8
# llama13b - 0.20 = 8, 8, 8, 8, 8, 8, 10, 2, 2, 16, 8, 2, 2, 2, 2, 4, 2, 2, 4, 2, 4, 2, 2, 10, 6, 10, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 10, 10, 8, 8
# llama13b - 0.50 = 8, 8, 8, 8, 8, 8, 10, 2, 2, 16, 2, 2, 2, 2, 8, 2, 2, 4, 4, 2, 4, 2, 12, 6, 4, 8, 2, 8, 10, 14, 8, 8, 8, 8, 8, 8, 8, 10, 8, 8
