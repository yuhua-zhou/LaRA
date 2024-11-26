import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from model.attention_fusion import FeatureFusion
from model.encoders import PositionalEncoder, LayerInfoEncoder, LayerPruneEncoder, BudgetEncoder


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_sized = output_size
        self.num_layers = 1

        # encoders
        self.pos_encoder = PositionalEncoder(num_hiddens=input_size)
        self.prune_encoder = LayerPruneEncoder(embed_size=input_size)
        self.info_encoder = LayerInfoEncoder(output_size=input_size)

        self.budget_encoder = BudgetEncoder(hidden_size=hidden_size)

        # feature fuser
        self.fuser = FeatureFusion(embedding_dim=input_size, feature_nums=2)

        self.attention = nn.Linear(hidden_size, hidden_size)

        # batch_first：默认为False。如果为True，则输入和输出的形状从(seq, batch, feature)调整为(batch, seq, feature)；
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.neck = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

        self._init_weights()

    def get_input_embedding(self, layer_info, prune_list):
        layer_encoding = self.info_encoder(layer_info)
        layer_encoding = layer_encoding.squeeze(2)
        prune_encoding = self.prune_encoder(prune_list)

        embedding = torch.stack([prune_encoding, layer_encoding], dim=2)
        embedding = self.fuser(embedding)
        embedding = self.pos_encoder(embedding)

        return embedding

    def forward(self, layer_info, prune_list, budget_list):
        x = self.get_input_embedding(layer_info, prune_list)
        X = self.att_scaled_dot_seq_len(x)
        batch_size, seq_length, _ = x.shape

        # h = self.budget_encoder(budget_list)  # hidden state = output state
        # h = h.unsqueeze(0).repeat(self.num_layers, 1, 1)

        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.double,
                        device=x.device)  # hidden state
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.double,
                        device=x.device)  # cell state

        x, _ = self.lstm(x, (h, c))

        logits = self.neck(x)
        return logits

    def att_scaled_dot_seq_len(self, x):
        # b, s, input_size / b, s, hidden_size
        # x = self.attention(x)  # bsh--->bst

        x = self.attention(x)

        score = torch.bmm(x, x.permute(0, 2, 1))  # bst*bts=bss
        score = score / np.sqrt(x.shape[2])
        attention = F.softmax(score, dim=-1)  # b s s
        context_vector = torch.bmm(attention, x)  # bss * bst ---> bst
        # context_vector = F.relu(context_vectore)

        return context_vector

    def _init_weights(self):
        # nn.init.kaiming_normal(self.rnn.weight, mode='fan_in', nonlinearity='relu')
        for layer in self.neck.modules():
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

        # 使用Kaiming初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.kaiming_normal_(param, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias_ih' in name or 'bias_hh' in name:
                nn.init.constant_(param, val=0)


class NASEnvironment:
    def __init__(self, action_space, predictor, alpha=1, beta=0, device="cpu"):
        super().__init__()
        self.action_space = action_space
        self.alpha = alpha
        self.beta = beta
        self.predictor = predictor
        self.device = device

    def step(self, actions, layer_info, prune_list, budget_list):
        ranks = actions.detach().to("cpu").apply_(lambda x: self.action_space[x])
        ranks = ranks.to(torch.float64).to(self.device)

        ranks_sum = torch.sum(ranks, dim=1).unsqueeze(1)

        print("ranks_sum: ", ranks_sum)
        print("budget_list: ", budget_list)

        ranks_sum = self.normalize_ranks(ranks_sum, ranks.shape[1])
        budget_list = self.normalize_ranks(budget_list, ranks.shape[1])

        penalty = ranks_sum - budget_list
        penalty = torch.where(penalty < 1, penalty.abs(), penalty ** 2)

        # evaluate_architecture
        performance = self.evaluate_architecture(ranks, layer_info, prune_list)
        performance = performance.unsqueeze(1)

        torch.set_printoptions(precision=8)
        print("performance: ", performance)
        # print("penalty: ", penalty)
        # penalty = torch.log(1 + penalty)
        # print("log penalty: ", penalty)

        return self.alpha * performance - self.beta * penalty

    def normalize_ranks(self, ranks, seq_len=1):
        min_rank = seq_len * self.action_space[0]
        max_rank = seq_len * self.action_space[-1]

        return 10 * (ranks - min_rank) / (max_rank - min_rank)

    def evaluate_architecture(self, rank_list, layer_info, prune_list):
        # construct a batch
        rank_list = rank_list.unsqueeze(2)

        # output as reward
        output = self.predictor(layer_info, rank_list, prune_list)
        performance = torch.sum(output, dim=1)
        return performance


class NASAgent:
    def __init__(self, policy_net, env, temperature=1):
        self.policy_net = policy_net
        self.env = env
        self.temperature_0 = temperature
        self.temperature = temperature

    def select_action(self, state, budget_list):
        layer_infos, prune_lists = state

        logits = self.policy_net(layer_infos, prune_lists, budget_list)
        probs = F.softmax(logits / self.temperature, dim=2)

        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()

        # 试一下log
        log_probs = dist.log_prob(actions)

        # actions = probs.multinomial(num_samples=1)
        return probs, log_probs, actions

    def update_policy(self, log_probs, rewards):
        discounted_reward = rewards / log_probs.shape[1]

        # discounted_reward相当于梯度，log_probs继承了计算神经网络中的参数梯度
        policy_loss = -(discounted_reward * log_probs).mean()

        # print("rewards:", rewards.shape)
        # print("log_probs: ", log_probs.shape)
        # print("rewards * log_probs: ", (rewards * log_probs).shape)
        # print("policy_loss:", policy_loss)

        return policy_loss

    def update_temperature(self, epoch):
        t = epoch / 10 + 1
        self.temperature = self.temperature_0 / (1 + math.log(t))


if __name__ == "__main__":
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # batch_size = 1
    # seq_length = 2
    # embed_size = 64
    #
    # prunes = [[0.5] for i in range(seq_length)]
    #
    # layer_info = torch.randn(batch_size, seq_length, 6, 4096).to(torch.double).to(device)
    # prune_list = torch.tensor([prunes for i in range(batch_size)]).to(torch.double).to(device)
    # budget_list = torch.tensor([[256] for i in range(batch_size)]).to(torch.double).to(device)
    #
    # policy_network = PolicyNetwork(embed_size, 128, 8).double().to(device)
    # result = policy_network(layer_info, prune_list, budget_list)
    #
    # print(result.shape)

    lstm = nn.LSTM(input_size=2, hidden_size=4, num_layers=1, batch_first=True)
    linear = nn.Linear(2, 4)
    neck = nn.Linear(4, 1)

    input = torch.ones(1, 4, 2)
    print(input)
    output, (h, c) = lstm(input)
    output = neck(output)
    print(h, c)
    print(output)

    output = linear(input)
    output = neck(output)
    print(output)
