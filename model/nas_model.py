import torch
import torch.nn as nn
import torch.nn.functional as F
from performance_preditor import PerformancePredictor


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
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def step(self, actions):
        ranks = [self.action_space[i] for i in actions.detach()]

        # 在这里评估架构性能,并返回奖励信号
        reward = self.evaluate_architecture(ranks)
        return reward

    def evaluate_architecture(self, ranks):
        print(ranks)
        print(sum(ranks))
        return torch.sum(torch.tensor(ranks))


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
        return logits, actions

    def update_policy(self, log_logits, rewards):
        discounted_rewards = rewards / log_logits.shape[0]

        policy_loss = -(discounted_rewards * log_logits).mean()
        print(policy_loss)

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


action_space = [2, 4, 6, 8, 10, 12, 14, 16]
num_episode = 10
num_epoch = 100

policy_net = PolicyNetwork(32, 64, len(action_space))
env = NASEnvironment(action_space)
agent = NASAgent(policy_net, env)

for epoch in range(num_epoch):
    for episode in range(num_episode):
        state = torch.zeros(10, 32)
        logits, actions = agent.select_action(state)
        reward = env.step(actions)

        log_logits = logits.gather(1, actions).squeeze(1)
        agent.update_policy(log_logits, reward)

# policy_net = PolicyNetwork(32, 64, len(search_space))
# dummy_input = torch.randn(32, 32)
# output = policy_net(dummy_input)
# print(output.shape)
# probs = F.softmax(output, dim=1)
# print(probs.shape)
# action = probs.multinomial(num_samples=1)
# print(action)
