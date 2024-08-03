import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_fusion import FeatureFusion
from .encoders import PositionalEncoder, LayerInfoEncoder, LayerPruneEncoder, BudgetEncoder


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

        self.budget_encoder = BudgetEncoder(hidden_size=hidden_size)

        # feature fuser
        self.fuser = FeatureFusion(embedding_dim=input_size, feature_nums=2)

        # batch_first：默认为False。如果为True，则输入和输出的形状从(seq, batch, feature)调整为(batch, seq, feature)；
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.neck = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

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

        batch_size, seq_length, _ = x.shape

        h = self.budget_encoder(budget_list)  # hidden state = output state
        h = h.unsqueeze(0).repeat(self.num_layers, 1, 1)

        # h = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.double,
        #                 device=x.device)  # hidden state
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.double,
                        device=x.device)  # cell state

        x, _ = self.rnn(x, (h, c))

        logits = self.neck(x)
        return logits


class NASEnvironment:
    def __init__(self, action_space, predictor, scale_factor=10, device="cpu"):
        super().__init__()
        self.action_space = action_space
        self.scale_factor = scale_factor
        self.predictor = predictor
        self.device = device

    def step(self, actions, layer_info, prune_list, budget_list):
        ranks = actions.detach().to("cpu").apply_(lambda x: self.action_space[x])
        ranks = ranks.to(torch.float64).to(self.device)

        ranks_sum = torch.sum(ranks, dim=1).unsqueeze(1)

        print(ranks_sum, budget_list)

        # ranks_sum = self.normalize_ranks(ranks_sum, ranks.shape[1])
        # budget_list = self.normalize_ranks(budget_list, ranks.shape[1])

        # evaluate_architecture
        # performance = self.evaluate_architecture(ranks, layer_info, prune_list)

        penalty = F.mse_loss(ranks_sum, budget_list)

        # penalty = -self.scale_factor * torch.log(1 + penalty)
        penalty = -self.scale_factor * penalty

        # return performance + self.scale_factor * penalty

        return penalty

    def normalize_ranks(self, ranks, seq_len=1):
        min_rank = seq_len * self.action_space[0]
        max_rank = seq_len * self.action_space[-1]

        return (ranks - min_rank) / (max_rank - min_rank)

    def evaluate_architecture(self, rank_list, layer_info, prune_list):
        # construct a batch
        rank_list = rank_list.unsqueeze(2)

        # output as reward
        output = self.predictor(layer_info, rank_list, prune_list)
        performance = torch.mean(output)

        torch.set_printoptions(precision=8)
        # print(output)
        # print(performance)

        return performance


class NASAgent:
    def __init__(self, policy_net, env, gamma=0.99, lr=1e-5):
        self.policy_net = policy_net
        self.env = env
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state, budget_list):
        layer_infos, prune_lists = state

        logits = self.policy_net(layer_infos, prune_lists, budget_list)
        probs = F.softmax(logits, dim=2)

        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()

        # 试一下log
        log_probs = dist.log_prob(actions)

        # actions = probs.multinomial(num_samples=1)
        return probs, log_probs, actions

    def update_policy(self, log_logits, rewards):
        discounted_rewards = rewards
        policy_loss = (discounted_rewards * log_logits).mean()
        print(policy_loss)

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    seq_length = 2
    embed_size = 64

    prunes = [[0.5] for i in range(seq_length)]

    layer_info = torch.randn(batch_size, seq_length, 6, 4096).to(torch.double).to(device)
    prune_list = torch.tensor([prunes for i in range(batch_size)]).to(torch.double).to(device)
    budget_list = torch.tensor([[256] for i in range(batch_size)]).to(torch.double).to(device)

    policy_network = PolicyNetwork(embed_size, 128, 8).double().to(device)
    result = policy_network(layer_info, prune_list, budget_list)

    print(result.shape)
