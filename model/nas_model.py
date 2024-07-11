import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import PositionalEncoder, LayerInfoEncoder, LayerPruneEncoder, LayerRankEncoder
from .attention_fusion import FeatureFusion


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        # self.rnn = nn.GRU(input_size, hidden_size)

        # encoders
        self.pos_encoder = PositionalEncoder(num_hiddens=input_size)
        self.prune_encoder = LayerPruneEncoder(embed_size=input_size)
        self.info_encoder = LayerInfoEncoder(output_size=input_size)

        # feature fuser
        self.fuser = FeatureFusion(embedding_dim=input_size, feature_nums=2)

        # batch_first：默认为False。如果为True，则输入和输出的形状从(seq, batch, feature)调整为(batch, seq, feature)；
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.neck = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
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
        x, _ = self.rnn(x)

        logits = self.neck(x)
        return logits

    def _init_weights(self):
        # nn.init.kaiming_normal(self.rnn.weight, mode='fan_in', nonlinearity='relu')
        for layer in self.neck.modules():
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

        # 使用Kaiming初始化
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.kaiming_normal_(param, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias_ih' in name or 'bias_hh' in name:
                nn.init.constant_(param, val=0)


class NASEnvironment:
    def __init__(self, action_space, predictor, scale_factor=1e2, device="cpu"):
        super().__init__()
        self.action_space = action_space
        self.scale_factor = scale_factor
        self.predictor = predictor
        self.device = device

    def step(self, actions, layer_info, prune_list):
        ranks = actions.detach().to("cpu").apply_(lambda x: self.action_space[x])
        ranks = torch.tensor(ranks, dtype=torch.float64, device=self.device)
        print(ranks)

        # evaluate_architecture
        reward = self.evaluate_architecture(ranks, layer_info, prune_list)
        return reward

    def evaluate_architecture(self, rank_list, layer_info, prune_list):
        n_layers = rank_list.shape[1]
        low = n_layers * self.action_space[0]
        high = n_layers * self.action_space[-1]
        penalty = (torch.sum(rank_list).detach() - low) / (high - low)
        penalty = torch.log(penalty)

        # construct a batch
        rank_list = rank_list.unsqueeze(2)

        # output as reward
        output = self.predictor(layer_info, rank_list, prune_list)
        reward = torch.mean(output)

        torch.set_printoptions(precision=8)
        print(output)
        return self.scale_factor * reward + penalty


class NASAgent:
    def __init__(self, policy_net, env, gamma=0.99, lr=1e-4):
        self.policy_net = policy_net
        self.env = env
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        layer_infors, prune_lists = state

        logits = self.policy_net(layer_infors, prune_lists)
        probs = F.softmax(logits, dim=2)

        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        # actions = actions.unsqueeze(1)

        print(probs.shape, dist, actions.shape)

        # 试一下log
        log_probs = dist.log_prob(actions)

        # actions = probs.multinomial(num_samples=1)
        return probs, log_probs, actions

    def update_policy(self, log_logits, rewards):
        discounted_rewards = rewards
        # discounted_rewards = rewards / log_logits.shape[0]
        policy_loss = -(discounted_rewards * log_logits).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    seq_length = 32

    prunes = [[0.5] for i in range(seq_length)]

    layer_info = torch.randn(batch_size, seq_length, 6, 4096).to(torch.double).to(device)
    prune_list = torch.tensor([prunes for i in range(batch_size)]).to(torch.double).to(device)

    policy_network = PolicyNetwork(64, 128, 8).double().to(device)
    result = policy_network(layer_info, prune_list)
    print(result.shape)
    print(result)
