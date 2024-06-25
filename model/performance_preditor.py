import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoder(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, max_len=1000):
        super(PositionalEncoder, self).__init__()

        # 创建位置编码矩阵
        self.pe = torch.zeros(max_len, num_hiddens)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_hiddens, 2).float() * (-math.log(10000.0) / num_hiddens))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        # 不需要梯度
        self.pe.requires_grad = False

    def forward(self, x):
        self.pe = self.pe.to(x.device)
        return x + self.pe[:x.size(1), :]


class LayerPruneEncoder(nn.Module):
    # def __init__(self, embed_size, max_len=1000):
    #     super(LayerPruneEncoder, self).__init__()
    #     self.P = torch.ones((max_len, embed_size))
    #
    # def forward(self, X):
    #     return self.P[:X.shape[0]] * X[:, None]

    def __init__(self, embed_size):
        super(LayerPruneEncoder, self).__init__()
        self.encode = nn.Linear(1, embed_size)

    def forward(self, x):
        return self.encode(x)


class LayerRankEncoder(nn.Module):
    # def __init__(self, embed_size, max_len=1000):
    #     super(LayerRankEncoder, self).__init__()
    #     self.P = torch.ones((1, max_len, embed_size))
    #
    # def forward(self, X):
    #     return self.P[:X.shape[0]] * X[:, None]

    def __init__(self, embed_size):
        super(LayerRankEncoder, self).__init__()
        self.encode = nn.Linear(1, embed_size)

    def forward(self, x):
        return self.encode(x)


class LayerInfoEncoder(nn.Module):
    def __init__(self, output_size):
        super(LayerInfoEncoder, self).__init__()
        self.down_sample = nn.AdaptiveAvgPool2d((1, output_size))
        self.encode = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self.down_sample(x)
        x = self.encode(x)
        return x


class PerformancePredictor(nn.Module):
    def __init__(self, prune_size=16, rank_size=16, layer_size=32,
                 hidden_size=128, num_layers=2, task_size=7):
        super(PerformancePredictor, self).__init__()

        self.input_size = prune_size + rank_size + layer_size

        self.pos_encoder = PositionalEncoder(num_hiddens=self.input_size).to("cuda:0")
        self.prune_encoder = LayerPruneEncoder(embed_size=prune_size)
        self.rank_encoder = LayerRankEncoder(embed_size=rank_size)
        self.info_encoder = LayerInfoEncoder(output_size=layer_size)

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        # self.attention = nn.Linear(64, t)
        self.output = nn.Linear(in_features=hidden_size, out_features=task_size)

    # 需要优化
    def get_input_embedding(self, layer_info, rank_list, prune_list):
        layer_encoding = self.info_encoder(layer_info)
        layer_encoding = layer_encoding.squeeze(2)
        rank_encoding = self.rank_encoder(rank_list)
        prune_encoding = self.prune_encoder(prune_list)

        embedding = torch.cat((prune_encoding, rank_encoding, layer_encoding), dim=2)
        embedding = self.pos_encoder(embedding)

        return embedding

    def forward(self, layer_info, rank_list, prune_list):
        # input x = [batch, seq_len, input_size]

        x = self.get_input_embedding(layer_info, rank_list, prune_list)
        x = self.att_scaled_dot_seq_len(x)
        x, _ = self.lstm(x)
        x = self.output(x)
        return x[:, -1, :]

    def att_scaled_dot_seq_len(self, x):
        # b, s, input_size / b, s, hidden_size
        # x = self.attention(x)  # bsh--->bst

        e = torch.bmm(x, x.permute(0, 2, 1))  # bst*bts=bss
        e = e / np.sqrt(x.shape[2])
        attention = F.softmax(e, dim=-1)  # b s s
        out = torch.bmm(attention, x)  # bss * bst ---> bst
        out = F.relu(out)

        return out


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    seq_length = 32

    ranks = [[2] for i in range(seq_length)]
    prunes = [[0.5] for i in range(seq_length)]

    layer_info = torch.randn(batch_size, seq_length, 6, 4096).to(torch.double).to(device)
    rank_list = torch.tensor([ranks for i in range(batch_size)]).to(torch.double).to(device)
    prune_list = torch.tensor([prunes for i in range(batch_size)]).to(torch.double).to(device)

    predictor = PerformancePredictor().double().to(device)
    result = predictor(layer_info, rank_list, prune_list)
    print(result)
