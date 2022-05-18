import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerDecoderLayer
from .positional_embedding import build_positional_embedding


class CrossBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, nheads):
        super().__init__()
        self.cross_attn = TransformerDecoderLayer(
            d_model=q_dim, nhead=nheads, dim_feedforward=4*q_dim, activation='gelu'
        )
        if q_dim == kv_dim:
            self.proj = None
        else:
            self.proj = nn.Linear(kv_dim, q_dim)

    def forward(self, q, kv, pos_embed_q, pos_embed_kv):
        """
        q & kv & pos_embed_q & pos_embed_kv: [hw, B, C]
        """
        if self.proj is not None:
            kv = self.proj(kv)
        return self.cross_attn(
            tgt=q, memory=kv,
            query_pos=pos_embed_q if pos_embed_q is not None else None,
            pos=pos_embed_kv if pos_embed_kv is not None else None
        )


class ViTPyramid(nn.Module):
    def __init__(self, reductions, hidden_dims, num_cross,
                 nheads=8, pos_embed_type='sine', image_size=224):
        super().__init__()

        self.patch_embeds = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        self.cross_blocks = nn.ModuleList()
        self.pos_embeds = nn.ModuleList()
        for i in range(num_cross):
            self.patch_embeds.append(nn.Conv2d(in_channels=3, out_channels=hidden_dims[i],
                        kernel_size=reductions[i], stride=reductions[i])
            )
            self.layer_norm.append(nn.LayerNorm(hidden_dims[i]))
            self.cross_blocks.append(CrossBlock(
                q_dim=hidden_dims[i], kv_dim=hidden_dims[-1], nheads=nheads
            ))
            self.pos_embeds.append(build_positional_embedding(
                type=pos_embed_type,
                shape=(image_size//reductions[i], image_size//reductions[i]),
                hidden_dim=hidden_dims[i]
            ))
    
    def forward(self, imgs, fs):
        v_feature_list = []
        for i in range(len(self.patch_embeds)):
            c = self.patch_embeds[i](imgs)
            ori_shape = c.shape
            c = c.flatten(2).permute(2, 0, 1)   # [B, C, h, w] -> [hw, B, C]
            c = self.layer_norm[i](c)
            c = self.cross_blocks[i](
                q=c, kv=fs[i].flatten(2).permute(2, 0, 1),
                pos_embed_q=self.pos_embeds[i](imgs.shape[0]).permute(1, 0, 2).to(c.device),
                pos_embed_kv=None   # kv already containing positional features
            ).permute(1, 2, 0).view(ori_shape)
            v_feature_list.append(c)
        v_feature_list.append(fs[-1])
        return v_feature_list


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
