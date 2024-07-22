import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .encoders import PositionalEncoder, LayerInfoEncoder, LayerPruneEncoder, LayerRankEncoder
from .attention_fusion import FeatureFusion


class PerformancePredictor(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_tasks=7):
        super(PerformancePredictor, self).__init__()

        # encoders
        self.pos_encoder = PositionalEncoder(num_hiddens=input_size)
        self.prune_encoder = LayerPruneEncoder(embed_size=input_size)
        self.rank_encoder = LayerRankEncoder(embed_size=input_size)
        self.info_encoder = LayerInfoEncoder(output_size=input_size)

        # feature fuser
        self.fuser = FeatureFusion(embedding_dim=input_size)

        # lstm
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=False)
        # self.attention = nn.Linear(64, t)

        # neck
        self.neck = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU()
        )

        # task heads
        self.task_heads = nn.ModuleList([
            nn.Linear(in_features=hidden_size, out_features=1) for _ in range(num_tasks)
        ])

        self.output = nn.Linear(in_features=hidden_size, out_features=num_tasks)

    # 需要优化
    def get_input_embedding(self, layer_info, rank_list, prune_list):
        layer_encoding = self.info_encoder(layer_info)
        layer_encoding = layer_encoding.squeeze(2)
        rank_encoding = self.rank_encoder(rank_list)
        prune_encoding = self.prune_encoder(prune_list)

        embedding = torch.stack([prune_encoding, rank_encoding, layer_encoding], dim=2)
        embedding = self.fuser(embedding)
        embedding = self.pos_encoder(embedding)

        return embedding

    def forward(self, layer_info, rank_list, prune_list):
        # input x = [batch, seq_len, input_size]

        x = self.get_input_embedding(layer_info, rank_list, prune_list)
        x = self.att_scaled_dot_seq_len(x)
        x, _ = self.lstm(x)

        # 只取lstm最后一层输出
        x = x[:, -1, :]
        x = self.neck(x)

        output = torch.cat([
            task_head(x) for task_head in self.task_heads
        ], dim=1)

        return output

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
    print(result.shape)
    print(result)
