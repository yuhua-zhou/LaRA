import torch.nn as nn
import torch
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
