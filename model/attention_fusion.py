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

if __name__ == "__main__":
    batch_size = 2
    embedding_dim = 64
    seq_len = 1

    prune = torch.rand(batch_size, seq_len, embedding_dim)
    info = torch.rand(batch_size, seq_len, embedding_dim)
    rank = torch.rand(batch_size, seq_len, embedding_dim)

    fusion_feature = torch.stack([prune, info, rank], dim=2)
    print(fusion_feature.shape)
    # print(fusion_feature)

    fuser = FeatureFusion(embedding_dim)
    fusion_feature = fuser(fusion_feature)
    print(fusion_feature.shape)
