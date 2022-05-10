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


class JointModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.v_backbone = build_v_backbone(cfg.backbone_key)
        self.l_backbone = RoBERTa()
        self.l_proj = nn.Linear(cfg.l_backbone.hidden_dim, cfg.hidden_dim)
        self.decoder = build_decoder(cfg.task.decoder)
        self.initialize(cfg.v_backbone.fix)

        self.criterion = build_loss(cfg.task.loss)
    
    def initialize(self, fix_v_backbone=True):
        for p in self.l_backbone.parameters():
            p.requires_grad_(False)
        if fix_v_backbone:
            for p in self.v_backbone.parameters():
                p.requires_grad_(False)

    def forward(self, imgs, txts=None):
        """
        intermediate tensors are required to be inferred and stored in buffer_path offline
        buffer_names: stacked string-ids of data samples
        """
        self.device = next(self.parameters()).device
        
        if txts is not None:
            txt_seqs, txt_pad_masks = self.encode_txt(txts)
            txt_seqs = self.l_proj(txt_seqs)
        #  [B, T, D],  [B, T]
        
        v_feature_list = self.v_backbone(imgs)

        return self.decoder(v_feature_list, txt_seqs, txt_pad_masks)

    @torch.no_grad()
    def encode_txt(self, txts):
        txt_seqs, token_inputs = self.l_backbone(txts, device=self.device)
        txt_pad_masks = token_inputs['attention_mask'].to(torch.bool)   # 0(False) for pad
        # txt_seqs: [B, T, D]
        # txt_pad_masks: [B, T]
        return txt_seqs, txt_pad_masks
