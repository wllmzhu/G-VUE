import torch
import torch.nn as nn
from .transformer import build_transformer_encoder, build_transformer_decoder
from .positional_embedding import build_positional_embedding
from fvcore.common.registry import Registry
DECODER = Registry('Decoder')


@DECODER.register()
class QueryType(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.v_proj = []
        for input_dim in cfg.input_dim_list:
            self.v_proj.append(nn.Linear(input_dim, cfg.hidden_dim))
        
        self.block_1 = build_transformer_encoder(cfg.transformer_encoder)
        self.block_2 = build_transformer_decoder(cfg.transformer_decoder)

        self.pos_embed = build_positional_embedding(cfg.positional_embedding)
        self.query_embed = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
    
    def forward(self, v_feature_list, txt_seqs=None, txt_pad_masks=None, train=True):
        img_seqs = v_feature_list[-1]   # get last feature
        B, C, h, w = img_seqs.shape
        
        # [B, C, h, w] -> [hw, B, C]
        img_seqs = img_seqs.flatten(2).permute(2, 0, 1)
        img_seqs = self.v_proj[-1](img_seqs)

        img_masks = torch.ones((h*w, B))

        if txt_seqs is not None:
            txt_seqs = txt_seqs.permute(1, 0, 2)   # [B, T, C] -> [T, B, C]
            img_seqs = torch.cat([img_seqs, txt_seqs], dim=0)
            img_masks = torch.cat([img_masks, txt_pad_masks], dim=1)
            pos_embed = torch.cat([self.pos_embed.weight, torch.zeros_like(txt_seqs)], dim=0)
        
        memory = self.block_1(src=img_seqs, src_key_padding_mask=img_masks, pos=pos_embed)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        hs = self.block_2(
            tgt=query_embed, memory=memory,
            memory_key_padding_mask=img_masks, pos=pos_embed, query_pos=query_embed
        )

        if train:
            pass
        else:
            pass


@DECODER.register()
class LabelType(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, joint_seqs):
        pass


@DECODER.register()
class DenseType(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, joint_seqs):
        pass


def build_decoder(decoder_cfg):
    assert cfg.key in ['QueryType', 'LabelType', 'DenseType']
    return DECODER.get(cfg.key)(cfg.params)
