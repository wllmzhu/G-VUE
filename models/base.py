"""
The whole VisualModel is composed of three parts: v_backbone + l_backbone + task_decoder
l_backbone should be fixed, since it is not relevant to visual representation
v_backbone is preferred to be fixed, but finetuning is also ok, by modifying cfg.v_backbone.fix to False
task_decoder is supposed to be trained, from which we evaluate visual represenation
"""
import torch
import torch.nn as nn
import numpy as np

from .v_backbone import build_v_backbone
from .l_backbone import RoBERTa
from .decoder import build_decoder
from .loss import build_loss
from .decoder_utils import ViTPyramid


class JointModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.v_backbone = build_v_backbone(cfg.v_backbone)
        self.l_backbone = RoBERTa(cache_dir=cfg.l_backbone.cfg_dir)
        self.l_proj = nn.Linear(cfg.l_backbone.hidden_dim, cfg.hidden_dim)
        self.decoder = build_decoder(cfg.task.decoder)

        # if 'ViT' in cfg.v_backbone.key and cfg.task.decoder.key == 'DenseType':
        #     # self.vit_pyramid = ViTPyramid(
        #     #     reductions=cfg.v_backbone.reduction,
        #     #     hidden_dims=cfg.v_backbone.hidden_dim,
        #     #     num_cross=len(cfg.v_backbone.extract_layer)-1
        #     # )
        #     self.vit_pyramid = nn.ModuleList(
        #         [
        #             nn.Sequential(
        #                 nn.ConvTranspose2d(cfg.v_backbone.hidden_dim[0], cfg.v_backbone.hidden_dim[0], 2, 2),
        #                 nn.GroupNorm(32, cfg.v_backbone.hidden_dim[0]),
        #                 nn.GELU(),
        #                 nn.ConvTranspose2d(cfg.v_backbone.hidden_dim[0], cfg.v_backbone.hidden_dim[0], 2, 2),
        #             ),
        #             nn.ConvTranspose2d(cfg.v_backbone.hidden_dim[1], cfg.v_backbone.hidden_dim[1], 2, 2),
        #             nn.Identity(),
        #             nn.MaxPool2d(2),
        #         ]
        #     )
        # else:
        #     self.vit_pyramid = None

        self.initialize(cfg.v_backbone.fix)

        self.criterion = build_loss(cfg.task.loss)
    
    def initialize(self, fix_v_backbone=True):
        for p in self.l_backbone.parameters():
            p.requires_grad_(False)
        if fix_v_backbone:
            for p in self.v_backbone.parameters():
                p.requires_grad_(False)

    def forward(self, imgs, txts=None):
        self.device = next(self.parameters()).device
        imgs = imgs.to(self.device)
        
        if txts is not None:
            txt_seqs, txt_pad_masks = self.encode_txt(txts)
            txt_seqs = self.l_proj(txt_seqs)
        #  [B, T, D],  [B, T]
        else:
            txt_seqs = None
            txt_pad_masks = None
        
        v_feature_list = self.v_backbone(imgs)

        # if self.vit_pyramid is not None:
        #     # v_feature_list = self.vit_pyramid(imgs, v_feature_list)
        #     v_feature_list = [self.vit_pyramid[i](f) for i, f in enumerate(v_feature_list)]

        # assume 'channel' lies ahead of 'shape' in visual features
        # [B, C, h, w] for CNNs, as well as for ViTs
        return self.decoder(v_feature_list, txt_seqs, txt_pad_masks)

    @torch.no_grad()
    def encode_txt(self, txts):
        txt_seqs, token_inputs = self.l_backbone(txts, device=self.device)
        txt_pad_masks = token_inputs['attention_mask'].to(torch.bool)   # 0(False) for pad
        # txt_seqs: [B, T, D]
        # txt_pad_masks: [B, T]
        return txt_seqs, txt_pad_masks
