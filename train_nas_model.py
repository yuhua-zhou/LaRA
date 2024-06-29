import json
import torch
from model.performance_preditor import PerformancePredictor, PositionalEncoder
from model.nas_model import PolicyNetwork, NASAgent, NASEnvironment
from utils.utils import read_layer_info


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


action_space = [2, 4, 6, 8, 10, 12, 14, 16]
num_epoch = 10
num_episode = 10

# 获取编码器
predictor = PerformancePredictor().double()
predictor.load_state_dict(torch.load("./output/performance_weights.pth"))
pos_encoder = PositionalEncoder(num_hiddens=48)

policy_net = PolicyNetwork(48, 64, len(action_space)).double()
env = NASEnvironment(action_space, predictor)
agent = NASAgent(policy_net, env)

# 模型设置
model_settings = get_model_setting("./dataset/merged_file_revise.json")
# print(model_settings)

for model in model_settings:
    name = model["name"]
    layer_info = torch.tensor(model["layer_info"], dtype=torch.float64)
    prune_list = torch.tensor(model["prune_list"], dtype=torch.float64)
    prune_list = prune_list.unsqueeze(1)

    layer_encoding = predictor.info_encoder(layer_info)
    layer_encoding = layer_encoding.squeeze(1)
    prune_encoding = predictor.prune_encoder(prune_list)

    # print(layer_encoding.shape)
    # print(prune_encoding.shape)

    embedding = torch.cat((prune_encoding, layer_encoding), dim=1)
    embedding = pos_encoder(embedding.unsqueeze(0))
    embedding = embedding.squeeze(0)

    # print(embedding.shape)

    state = embedding

    for epoch in range(num_epoch):
        for episode in range(num_episode):
            logits, actions = agent.select_action(state)
            reward = env.step(actions, layer_info, prune_list)

            log_logits = logits.gather(dim=1, index=actions).squeeze(1)
            loss = agent.update_policy(log_logits, reward)

            print("model %s: epoch: %d, episode: %d, reward: %s, loss: %s" % (name, epoch, episode, reward.detach().item(), loss.detach().item()))
            # print("model %s: epoch: %d, episode: %d, reward: %s, loss: %s" % (name, epoch, episode, reward, loss))
