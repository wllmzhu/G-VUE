import torch
import torch.nn as nn
from torch.nn import functional as F
from .transformer import build_transformer_encoder
from .positional_embedding import build_positional_embedding
from fvcore.common.registry import Registry
from mmcv.cnn import ConvModule
from models.decoder_utils import LabelMLP, DenseMLP, resize

import hydra


DECODER = Registry('Decoder')


@DECODER.register()
class LabelType(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.v_proj = nn.Linear(cfg.input_dim_list[-1], cfg.hidden_dim)
        
        self.block_1 = build_transformer_encoder(cfg.transformer_encoder)

        # LabelMLP is a MLP written for the LabelType decoder
        self.output_head = LabelMLP(cfg.transformer_encoder.hidden_dim, cfg.num_classes)

        self.pos_embed = build_positional_embedding(
            type=cfg.positional_embedding.type,
            shape=(224//cfg.reduction[-1], 224//cfg.reduction[-1]),
            hidden_dim=cfg.positional_embedding.hidden_dim
        )
        self.label_token = nn.Parameter(0.1*torch.randn(cfg.hidden_dim))
    
    def forward(self, v_feature_list, txt_seqs=None, txt_pad_masks=None):
        img_seqs = v_feature_list[-1]   # get last feature
        B, C, h, w = img_seqs.shape
        
        # [B, C, h, w] -> [hw, B, C]
        img_seqs = img_seqs.flatten(2).permute(2, 0, 1)
        img_seqs = self.v_proj(img_seqs)

        label_token = self.label_token.reshape(1, 1, -1).repeat(1, B, 1)
        img_seqs = torch.cat([label_token, img_seqs], dim=0)

        img_masks = torch.zeros((B, 1+h*w), dtype=int).to(self.label_token.device)
        pos_embed = self.pos_embed(B).permute(1, 0, 2).to(self.label_token.device)
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
    """
    SegFormer Decoder
    """
    def __init__(self, info):
        super().__init__()

        self.info = info
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.info.in_channels
        embed_dim = self.info.embed_dim

        # Dense is a MLP written for the DenseType decoder
        self.linear_c4 = DenseMLP(input_dim=c4_in_channels, embed_dim=embed_dim)
        self.linear_c3 = DenseMLP(input_dim=c3_in_channels, embed_dim=embed_dim)
        self.linear_c2 = DenseMLP(input_dim=c2_in_channels, embed_dim=embed_dim)
        self.linear_c1 = DenseMLP(input_dim=c1_in_channels, embed_dim=embed_dim)

        self.linear_fuse = ConvModule(
            in_channels=embed_dim*4,
            out_channels=embed_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.dropout = nn.Dropout2d(self.info.dropout_ratio)
        self.linear_pred = nn.Conv2d(embed_dim, self.info.num_classes, kernel_size=1)


    def forward(self, v_feature_list):
        c1, c2, c3, c4 = v_feature_list

        ############## MLP decoder on C1-C4 ###########
        n, _, _, _ = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear')

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear')

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear')

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        
        x = resize(x, size=self.info.output_resolution, mode='bilinear')

        return x



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


@hydra.main(config_path='../configs/decoder', config_name='DenseType')
def main(cfg):
    dense = DenseType(cfg.info)
    in_channels = [256, 512, 1024, 2048]
    r = [int(224/4), int(224/8), int(224/16), int(224/32)]
    
    temp = [torch.randn((8,in_channels[0],r[0],r[0])),
            torch.randn((8,in_channels[1],r[1],r[1])),
            torch.randn((8,in_channels[2],r[2],r[2])),
            torch.randn((8,in_channels[3],r[3],r[3]))]

    temp = [tmp.cuda() for tmp in temp]
    dense.cuda()

    x = dense(temp)
    print(x.shape)

if __name__ == '__main__':
    main()
