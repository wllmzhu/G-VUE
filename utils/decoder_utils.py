import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        #[B C H W] -> [B HW C]
        x = x.flatten(2).transpose(1, 2)
        #[B HW C] -> [B HW C_emb]
        x = self.proj(x)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)