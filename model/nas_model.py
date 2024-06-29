import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        logits = self.fc(x)
        return logits


class NASEnvironment:
    def __init__(self, action_space, predictor, scale_factor=100):
        super().__init__()
        self.action_space = action_space
        self.scale_factor = scale_factor
        self.predictor = predictor

    def step(self, actions, layer_info, prune_list):
        ranks = [self.action_space[i] for i in actions.detach()]
        print(ranks)
        ranks = torch.tensor(ranks, dtype=torch.float64).unsqueeze(1)

        # 在这里评估架构性能,并返回奖励信号
        reward = self.evaluate_architecture(ranks, layer_info, prune_list)
        return reward

    def evaluate_architecture(self, rank_list, layer_info, prune_list):
        rank_list = rank_list.unsqueeze(0)
        layer_info = layer_info.unsqueeze(0)
        prune_list = prune_list.unsqueeze(0)

        # print(rank_list.shape)
        # print(layer_info.shape)
        # print(prune_list.shape)

        output = self.predictor(layer_info, rank_list, prune_list)
        reward = torch.sum(output)

        return self.scale_factor * reward - torch.sum(torch.tensor(rank_list))


class NASAgent:
    def __init__(self, policy_net, env, gamma=0.99, lr=1e-3):
        self.policy_net = policy_net
        self.env = env
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        logits = self.policy_net(state)
        probs = F.softmax(logits, dim=1)
        actions = probs.multinomial(num_samples=1)
        return probs, actions

    def update_policy(self, log_logits, rewards):
        discounted_rewards = rewards / log_logits.shape[0]

        policy_loss = -(discounted_rewards * log_logits).mean()
        # print(policy_loss)

        self.optimizer.zero_grad()
        # 为什么需要加这句话？ 之前好像都不需要？
        policy_loss.backward(retain_graph=True)
        self.optimizer.step()

        return policy_loss
