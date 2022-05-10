"""
For visual representation in the form of grid features.
Unnecessary for language parts, since language backbone incorporates positional information already.
"""
import math
import torch
from torch import nn


class PositionalEmbeddingSine(nn.Module):
    def __init__(self, shape, num_pos_feats, temperature=10000):
        super().__init__()
        h, w = shape
        scale = 2 * math.pi
        eps = 1e-6

        y_embed = torch.arange(1, h+1).view(h, 1).repeat(1, w)
        x_embed = torch.arange(1, w+1).view(1, w).repeat(h, 1)

        y_embed = y_embed / (y_embed[-1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, -1:] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        # [h, w, num_pos_feats]

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        # [h, w, num_pos_feats]

        self.embed = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)   # [hw, 2*num_pos_feats]
        self.embed.requires_grad_(False)

    def forward(self, bs):
        return self.embed.unsqueeze(0).repeat(bs, 1, 1)


class PositionalEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, shape, num_pos_feats):
        super().__init__()
        self.h, self.w = shape
        self.row_embed = nn.Embedding(self.h, num_pos_feats)
        self.col_embed = nn.Embedding(self.w, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        self.row_embed.requires_grad_(True)
        self.col_embed.requires_grad_(True)

    def forward(self, bs):
        y_emb = self.row_embed
        x_emb = self.col_embed

        pos = torch.cat([
            y_emb.unsqueeze(1).repeat(1, self.w, 1),
            x_emb.unsqueeze(0).repeat(self.h, 1, 1)
        ], dim=-1).flatten(0, 1).unsqueeze(0).repeat(bs, 1, 1)
        return pos


def build_positional_embedding(type, shape, hidden_dim):
    N_steps = hidden_dim // 2
    if type == 'sine':
        pos_embed = PositionalEmbeddingSine(shape, N_steps)
    elif type == 'learned':
        pos_embed = PositionalEmbeddingLearned(shape, N_steps)
    else:
        raise ValueError(f"not supported {type}")

    return pos_embed
