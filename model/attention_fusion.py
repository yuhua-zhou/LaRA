import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    def __init__(self, embedding_dim, feature_nums=3, num_heads=8):
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.attn = nn.MultiheadAttention(embedding_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim * feature_nums, embedding_dim)
        self.relu = nn.ReLU()

        self._init_weights()

    def forward(self, x):
        batch_size, seq_len, feat_len, feat_embed = x.shape

        queries = self.query(x)
        queries = queries.view(batch_size * seq_len, feat_len, feat_embed)

        keys = self.key(x)
        keys = keys.view(batch_size * seq_len, feat_len, feat_embed)

        values = self.value(x)
        values = values.view(batch_size * seq_len, feat_len, feat_embed)

        output, _ = self.attn(queries, keys, values)

        # output = x

        output = output.view(batch_size, seq_len, feat_len, feat_embed)
        output = self.layer_norm(output)
        output = output.view(batch_size, seq_len, -1)
        output = self.fc(output)
        # output = self.relu(output)
        return output

    def _init_weights(self):
        # nn.init.kaiming_normal(self.rnn.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.query.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.key.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.value.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')

if __name__ == "__main__":
    batch_size = 2
    embedding_dim = 8
    seq_len = 1

    # prune = torch.rand(batch_size, seq_len, embedding_dim)
    # info = torch.rand(batch_size, seq_len, embedding_dim)
    # rank = torch.rand(batch_size, seq_len, embedding_dim)

    prune = torch.ones((1, seq_len, embedding_dim)) * 0.25
    info = torch.ones((1, seq_len, embedding_dim))
    rank = torch.ones((1, seq_len, embedding_dim))

    prune_0 = torch.ones((1, seq_len, embedding_dim)) * 0.3
    info_0 = torch.ones((1, seq_len, embedding_dim))
    rank_0 = torch.ones((1, seq_len, embedding_dim))

    fuser = FeatureFusion(embedding_dim)

    fusion_feature = torch.stack([prune, info, rank], dim=2)
    print(fusion_feature.shape)
    print(fusion_feature)
    fusion_feature = fuser(fusion_feature)
    print(fusion_feature.shape)
    print(fusion_feature)

    fusion_feature = torch.stack([prune_0, info_0, rank_0], dim=2)
    print(fusion_feature.shape)
    print(fusion_feature)
    fusion_feature = fuser(fusion_feature)
    print(fusion_feature.shape)
    print(fusion_feature)
