import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelMLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer1 = nn.Linear(in_channel, 2*in_channel)
        self.activation = F.gelu
        self.layernorm = nn.LayerNorm(2*in_channel)
        self.layer2 = nn.Linear(2*in_channel, out_channel)

    def forward(self, x):
        return self.layer2(self.layernorm(self.activation(self.layer1(x))))


class DenseMLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # [B C H W] -> [B HW C]
        x = x.flatten(2).transpose(1, 2)
        # [B HW C] -> [B HW C_emb]
        x = self.proj(x)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    elif isinstance(size, int):
        size = tuple([size, size])
    return F.interpolate(input, size, scale_factor, mode, align_corners)
