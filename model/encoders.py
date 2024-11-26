import math

import torch
import torch.nn as nn

from utils.utils import normalize

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
        self.encode = torch.ones((1, embed_size))
        # self.encode = nn.Linear(1, embed_size)

    def forward(self, x):
        # x = self.encode(x)
        # return x

        self.encode = self.encode.to(x.device)
        x = x * self.encode
        return x

class LayerRankEncoder(nn.Module):
    # def __init__(self, embed_size, max_len=1000):
    #     super(LayerRankEncoder, self).__init__()
    #     self.P = torch.ones((1, max_len, embed_size))
    #
    # def forward(self, X):
    #     return self.P[:X.shape[0]] * X[:, None]

    def __init__(self, embed_size):
        super(LayerRankEncoder, self).__init__()
        self.encode = torch.ones((1, embed_size))

        # self.encode = nn.Linear(1, embed_size)

    def forward(self, x):
        # x = (x - 2.0) / 14.0
        # x = self.encode(x)
        # return x

        self.encode = self.encode.to(x.device)
        x = (x - 2.0) / 14.0
        x = self.encode * x
        return x

class LayerInfoEncoder(nn.Module):
    # SVD
    def __init__(self, output_size):
        super(LayerInfoEncoder, self).__init__()

        self.hidden = output_size
        self.down_sample = nn.AdaptiveAvgPool1d(output_size)

        # self.encode = nn.Sequential(
        #     nn.Linear(output_size, output_size),
        #     # nn.Dropout(0.05),
        #     # nn.ReLU(),
        #     # nn.Linear(output_size, output_size)
        # )

    def forward(self, x):
        # x = x[:, :, :, :self.hidden].clone()
        # batch, seq_len, weight_num, hidden = x.shape
        #
        # x = x.view(batch, seq_len, weight_num * hidden)
        # x = self.down_sample(x)

        x = x.mean(dim=2)
        x = self.down_sample(x)

        # hidden = 64
        # x = normalize(x, mean=3.667495956768107, std=1.6198784927662087, max=22.479613939921062, min=1.3181705474853516)
        # hidden = 128
        x = normalize(x, mean=3.275902493126826, std=1.3711524055383268, max=22.479613939921062, min=1.1682074268658955)

        # x = self.encode(x)
        return x

    # PCA
    # def __init__(self, output_size):
    #     super(LayerInfoEncoder, self).__init__()
    #
    #     self.hidden = output_size
    #     self.down_sample = nn.AdaptiveAvgPool1d(output_size)
    #
    #     self.encode = nn.Sequential(
    #         nn.Linear(output_size, output_size),
    #     )
    #
    # def forward(self, x):
    #     batch, seq_len, weight_num, width, height = x.shape
    #
    #     x = x.view(batch, seq_len, -1)
    #     x = self.down_sample(x)
    #
    #     x = self.encode(x)
    #     return x

class BudgetEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, hidden_size // 2),
            # nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )

    def forward(self, x):
        return self.fc(x)

if __name__ == '__main__':
    # x = torch.randn(64, 32, 6, 4096)
    # encoder = LayerInfoEncoder(128)
    # print(encoder(x).shape)

    # x = torch.randn(64, 32, 6, 256, 256)
    # encoder = LayerInfoEncoder(128)
    # print(encoder(x).shape)

    x = torch.randn(1, 1)
    encoder = BudgetEncoder(hidden_size=64)
    x = encoder(x)
    print(x.shape)
