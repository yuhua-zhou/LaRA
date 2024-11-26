import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention_fusion import FeatureFusion
from model.encoders import PositionalEncoder, LayerInfoEncoder, LayerPruneEncoder

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_sized = output_size
        self.num_layers = 2

        # encoders
        self.pos_encoder = PositionalEncoder(num_hiddens=input_size)
        self.prune_encoder = LayerPruneEncoder(embed_size=input_size)
        self.info_encoder = LayerInfoEncoder(output_size=input_size)

        # feature fuser
        self.fuser = FeatureFusion(embedding_dim=input_size, feature_nums=2)

        self.attention = nn.Linear(input_size, input_size)

        # batch_first：默认为False。如果为True，则输入和输出的形状从(seq, batch, feature)调整为(batch, seq, feature)；
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True)

        self.neck = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, output_size)
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

    def forward(self, layer_info, prune_list):
        x = self.get_input_embedding(layer_info, prune_list)

        x = self.att_scaled_dot_seq_len(x)
        batch_size, seq_length, _ = x.shape

        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.double,
                        device=x.device)  # hidden state
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.double,
                        device=x.device)  # cell state

        x, _ = self.lstm(x, (h, c))
        logits = self.neck(x)

        # (batch, seq_len, 8)
        return logits

    def att_scaled_dot_seq_len(self, x):
        # b, s, input_size / b, s, hidden_size
        # [batch, seq_len, hidden]
        x = self.attention(x)  # bsh--->bst

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

class ReplayBuffer():
    def __init__(self, buffer_size):
        super().__init__()
        self.buffer_size = buffer_size
        self.buffer = []

    def push(self, item):
        self.buffer += item

        if self.full():
            evict_num = len(self.buffer) - self.buffer_size
            self.evict(evict_num)

    def evict(self, num):
        indices = range(len(self))
        weights = np.array([1.0 / torch.sum(reward).detach().cpu() for (_, _, _, _, reward) in self.buffer])
        weights_sum = np.sum(weights)
        weights = weights / weights_sum

        indices_to_remove = random.choices(indices, weights=weights, k=num)

        # indices_to_remove = random.sample(indices, num)

        self.buffer = [
            item for i, item in enumerate(self.buffer) if i not in indices_to_remove
        ]

    def sample(self, batch_size):
        batch_size = min(len(self), batch_size)
        samples = random.sample(self.buffer, batch_size)
        names, layer_infos, prune_lists, action_lists, rewards = zip(*samples)
        return torch.stack(layer_infos), torch.stack(prune_lists), torch.stack(action_lists), torch.stack(rewards)

    def full(self):
        return self.__len__() >= self.buffer_size

    def __len__(self):
        return len(self.buffer)

class NASEnvironment:
    def __init__(self, action_space, predictor, alpha=10, beta=0, device="cpu"):
        super().__init__()
        self.action_space = action_space
        self.alpha = alpha
        self.beta = beta
        self.predictor = predictor
        self.device = device

    def step(self, actions, layer_info, prune_list, budget_list=None):
        ranks = torch.clone(actions)
        ranks = ranks.detach().to("cpu").apply_(lambda x: self.action_space[x])
        ranks = ranks.to(torch.float64).to(self.device)

        penalty = 0.0

        result = []

        if budget_list is not None:
            ranks_sum = torch.sum(ranks, dim=1).unsqueeze(1)

            # print("ranks_sum: ", ranks_sum)
            # print("budget_list: ", budget_list)

            result.append(budget_list)
            result.append(ranks_sum)

            ranks_sum = self.normalize_ranks(ranks_sum, ranks.shape[1])  # [batch, 1]
            budget_list = self.normalize_ranks(budget_list, ranks.shape[1])  # [batch, 1]

            penalty = ranks_sum - budget_list
            penalty = torch.where(penalty < 0, penalty.abs(), (penalty + 1) ** 2)
            penalty = torch.log(1 + penalty)

            result.append(self.beta * penalty)

        # evaluate_architecture
        performance = self.evaluate_architecture(ranks, layer_info, prune_list)
        performance = performance.unsqueeze(1)

        # torch.set_printoptions(precision=8)
        # print("performance: ", performance)
        # print("penalty: ", penalty)
        # print("\n\n")

        result.append(self.alpha * performance)
        reward = self.alpha * performance - self.beta * penalty
        result.append(reward)

        return result

    def normalize_ranks(self, ranks, seq_len=1):
        min_rank = seq_len * self.action_space[0]
        max_rank = seq_len * self.action_space[-1]

        return 10 * (ranks - min_rank) / (max_rank - min_rank)

    def evaluate_architecture(self, rank_list, layer_info, prune_list):
        # construct a batch
        rank_list = rank_list.unsqueeze(2)

        # 7 * [max, min, mean, std]
        baseline = torch.tensor([
            [0.6832, 0.43476430976430974, 0.5363381516452773, 0.08248993406220569],
            [0.3882252559726962, 0.2354948805460751, 0.29954880463636213, 0.054358256949388895],
            [0.644, 0.494869771112865, 0.5615874467626749, 0.05026910104203764],
            [0.416, 0.208, 0.3066297101449275, 0.08522062323495214],
            [0.6743, 0.40275229357798165, 0.5500164683331117, 0.06737887878042395],
            [0.7758, 0.6653971708378672, 0.7158668326473324, 0.03994666480630215],
            [0.6935869348735312, 0.3569010157339175, 0.4956095242974675, 0.14156553183549178]
        ])

        with torch.no_grad():
            # output as reward
            output = self.predictor(layer_info, rank_list, prune_list)  # [batch, metric_num, value]

            # normalize the output
            # batch_size, metric_num = output.shape
            # for i in range(batch_size):
            #     for j in range(metric_num):
            #         mx, mn, mean, std = baseline[j]
            #         # output[i, j] = (output[i, j] - mean) / std
            #         output[i, j] = (output[i, j] - mn) / (mx - mn)

            performance = torch.mean(output, dim=1)
        return performance

class NASAgent:
    def __init__(self, policy_net, temperature=1):
        self.policy_net = policy_net
        self.temperature = temperature

    def select_action(self, state):
        layer_infos, prune_lists = state

        logits = self.policy_net(layer_infos, prune_lists)
        probs = F.softmax(logits / self.temperature, dim=2)

        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()

        # 这边修改成前4层8， 后两层8
        actions = actions.tolist()
        actions = [[3, 3, 3, 3] + action[4:-2] + [3, 3] for action in actions]
        actions = torch.tensor(actions, device=logits.device)

        log_probs = dist.log_prob(actions)

        return probs, log_probs, actions

    def update_policy(self, log_probs, rewards):
        discounted_reward = rewards / log_probs.shape[1]

        # discounted_reward相当于梯度，log_probs继承了计算神经网络中的参数梯度
        policy_loss = -(discounted_reward * log_probs).mean()

        # print("rewards:", rewards.shape) # [batch_size, 1]
        # print("log_probs: ", log_probs.shape) # [batch_size, 32]
        # print("rewards * log_probs: ", (rewards * log_probs).shape) # [batch_size, 32]
        # print("policy_loss:", policy_loss)

        return policy_loss

    def set_temperature(self, temperature):
        self.temperature = temperature

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

    buffer = ReplayBuffer(10)
    for i in range(10):
        buffer.push([(str(i), torch.tensor([i]), torch.tensor([i]), torch.tensor([i]), torch.tensor([i]))])

    print(buffer.buffer)
    print(buffer.sample(2))

    for j in range(10):
        i = j + 10
        buffer.push([(str(i), torch.tensor([i]), torch.tensor([i]), torch.tensor([i]), torch.tensor([i]))])
        print(buffer.buffer)


