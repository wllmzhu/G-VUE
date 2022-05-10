import torch
import torch.nn as nn
from torch.nn import functional as F
from .transformer import build_transformer_encoder
from .positional_embedding import build_positional_embedding
from fvcore.common.registry import Registry
DECODER = Registry('Decoder')


@DECODER.register()
class LabelType(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.v_proj = []
        for input_dim in cfg.input_dim_list:
            self.v_proj.append(nn.Linear(input_dim, cfg.hidden_dim))
        
        self.block_1 = build_transformer_encoder(cfg.transformer_encoder)
        self.output_head = MLP(cfg.transformer_encoder.hidden_dim, cfg.num_classes)

        self.pos_embed = build_positional_embedding(
            cfg.positional_embedding.type, shape=None, hidden_dim=cfg.positional_embedding.hidden_dim
        )
        self.label_token = nn.Parameter(0.1*torch.randn(cfg.hidden_dim))
    
    def forward(self, v_feature_list, txt_seqs=None, txt_pad_masks=None):
        img_seqs = v_feature_list[-1]   # get last feature
        B, C, h, w = img_seqs.shape
        
        # [B, C, h, w] -> [hw, B, C]
        img_seqs = img_seqs.flatten(2).permute(2, 0, 1)
        img_seqs = self.v_proj[-1](img_seqs)

        label_token = self.label_token.reshape(1, 1, -1).repeat(1, B, 1)
        img_seqs = torch.cat([label_token, img_seqs], dim=0)

        img_masks = torch.zeros((B, 1+h*w), dtype=int)
        pos_embed = self.pos_embed(B).permute(1, 0, 2)
        pos_embed = torch.cat([
            torch.zeros_like(self.label_token).view(1, 1, -1).repeat(1, B, 1), pos_embed
        ], dim=0)

        if txt_seqs is not None:
            txt_seqs = txt_seqs.permute(1, 0, 2)   # [B, T, C] -> [T, B, C]
            img_seqs = torch.cat([img_seqs, txt_seqs], dim=0)
            txt_pad_masks = ~txt_pad_masks
            img_masks = torch.cat([img_masks, txt_pad_masks], dim=1)
            pos_embed = torch.cat([pos_embed, torch.zeros_like(txt_seqs)], dim=0)
        
        memory = self.block_1(src=img_seqs, src_key_padding_mask=img_masks, pos=pos_embed)
        label_output = memory[0]   # first token
        label_output = self.output_head(label_output)
        # [B, classes]
        return label_output


@DECODER.register()
class DenseType(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, v_feature_list, txt_seqs=None, txt_pad_masks=None):
        pass


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer1 = nn.Linear(in_channel, 2*in_channel)
        self.activation = F.gelu
        self.layernorm = nn.LayerNorm(2*in_channel)
        self.layer2 = nn.Linear(2*in_channel, out_channel)

    def forward(self, x):
        return self.layer2(self.layernorm(self.activation(self.layer1(x))))


def build_decoder(cfg):
    # assert cfg.key in ['QueryType', 'LabelType', 'DenseType']
    assert cfg.key in ['LabelType', 'DenseType']
    return DECODER.get(cfg.key)(cfg.params)


# @DECODER.register()
# class QueryType(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()

#         self.v_proj = []
#         for input_dim in cfg.input_dim_list:
#             self.v_proj.append(nn.Linear(input_dim, cfg.hidden_dim))
        
#         self.block_1 = build_transformer_encoder(cfg.transformer_encoder)
#         self.block_2 = build_transformer_decoder(cfg.transformer_decoder)

#         self.pos_embed = build_positional_embedding(
#             cfg.positional_embedding.type, shape=None, hidden_dim=cfg.positional_embedding.hidden_dim
#         )
#         self.query_embed = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
    
#     def forward(self, v_feature_list, txt_seqs=None, txt_pad_masks=None, train=True):
#         img_seqs = v_feature_list[-1]   # get last feature
#         B, C, h, w = img_seqs.shape
        
#         # [B, C, h, w] -> [hw, B, C]
#         img_seqs = img_seqs.flatten(2).permute(2, 0, 1)
#         img_seqs = self.v_proj[-1](img_seqs)

#         img_masks = torch.zeros((B, h*w), dtype=int)
#         pos_embed = self.pos_embed(B).permute(1, 0, 2)

#         if txt_seqs is not None:
#             txt_seqs = txt_seqs.permute(1, 0, 2)   # [B, T, C] -> [T, B, C]
#             img_seqs = torch.cat([img_seqs, txt_seqs], dim=0)
#             txt_pad_masks = ~txt_pad_masks
#             img_masks = torch.cat([img_masks, txt_pad_masks], dim=1)
#             pos_embed = torch.cat([pos_embed, torch.zeros_like(txt_seqs)], dim=0)
        
#         memory = self.block_1(src=img_seqs, src_key_padding_mask=img_masks, pos=pos_embed)

#         query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
#         hs = self.block_2(
#             tgt=torch.zeros_like(query_embed), memory=memory,
#             memory_key_padding_mask=img_masks, pos=pos_embed, query_pos=query_embed
#         ).permute(1, 0, 2)

#         # TO BE IMPLEMENTED
