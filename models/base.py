"""
The whole VisualModel is composed of three parts: v_backbone + l_backbone + task_decoder
l_backbone should be fixed, since it is not relevant to visual representation
v_backbone is preferred to be fixed, but finetuning is also ok, by modifying cfg.v_backbone.fix to False
task_decoder is supposed to be trained, from which we evaluate visual represenation
"""
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

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

        self.initialize()

        self.criterion = build_loss(cfg.task.loss)
    
    def initialize(self):
        for p in self.l_backbone.parameters():
            p.requires_grad_(False)
        if self.v_backbone.fix:
            for p in self.v_backbone.parameters():
                p.requires_grad_(False)

    def forward(self, imgs, txts=None, expand_batch=False, add_special_token=False):
        self.device = next(self.parameters()).device
        imgs = imgs.to(self.device)
        
        if txts is not None:
            txt_seqs, txt_pad_masks = self.encode_txt(txts, expand_batch, add_special_token)
            #  [B, T, D],  [B, T]
            txt_seqs = self.l_proj(txt_seqs)
        else:
            txt_seqs = None
            txt_pad_masks = None
        
        if expand_batch:
            if imgs.ndim == 4:
                r = txt_seqs.shape[0] // imgs.shape[0]
                imgs = imgs.unsqueeze(1).repeat(1, r, 1, 1, 1)
                imgs = rearrange(imgs, 'B r C H W -> (B r) C H W')
            elif imgs.ndim == 5:
                imgs = rearrange(imgs, 'B r C H W -> (B r) C H W')
        
        v_feature_list = self.v_backbone(imgs)

        # if self.vit_pyramid is not None:
        #     # v_feature_list = self.vit_pyramid(imgs, v_feature_list)
        #     v_feature_list = [self.vit_pyramid[i](f) for i, f in enumerate(v_feature_list)]

        # assume 'channel' lies ahead of 'shape' in visual features
        # [B, C, h, w] for CNNs, as well as for ViTs
        return self.decoder(v_feature_list, txt_seqs, txt_pad_masks)

    @torch.no_grad()
    def encode_txt(self, txts, expand_batch=False, add_special_token=False):
        if expand_batch:
            # for VL-Matching tasks
            pairs_batch = []
            attn_masks = []
            # type_ids = []   # RoBERTa removes NSP task, so type_ids are unnecessary
            max_len = 0
            if add_special_token:
                # for VCR dual-sentences
                for txt in txts:
                    # Q*1 + A*4
                    question, answers = txt[0], txt[1:]
                    question = self.l_backbone.tokenizer.convert_tokens_to_ids(
                        self.l_backbone.tokenizer.tokenize(question)
                    )
                    for answer in answers:
                        answer = self.l_backbone.tokenizer.convert_tokens_to_ids(
                            self.l_backbone.tokenizer.tokenize(answer)
                        )
                        pair = self.l_backbone.tokenizer.build_inputs_with_special_tokens(
                            question, answer
                        )
                        curr_len = len(pair)
                        pairs_batch.append(pair)
                        attn_masks.append([1]*curr_len)
                        # type_ids.append([0]*(len(question)+2) + [1]*(len(answer)+2))
                        max_len = max(max_len, curr_len)
            else:
                # for Flickr retrieval
                for txt in txts:
                    # each sample contains multiple captions
                    for cap in txt:
                        pair = self.l_backbone.tokenizer.encode(cap)   # <s> and </s> automatically added
                        curr_len = len(pair)
                        pairs_batch.append(pair)
                        attn_masks.append([1]*curr_len)
                        # type_ids.append([0]*(curr_len))   # single type
                        max_len = max(max_len, curr_len)
            
            # pad to same length
            for i in range(len(pairs_batch)):
                pairs_batch[i] = pairs_batch[i] + [0]*(max_len-len(pairs_batch[i]))
                attn_masks[i] = attn_masks[i] + [0]*(max_len-len(attn_masks[i]))
                # type_ids[i] = type_ids[i] + [0]*(max_len-len(type_ids[i]))

            pairs_batch_input = torch.as_tensor(pairs_batch, dtype=torch.int64, device=self.device)
            pairs_batch_mask = torch.as_tensor(attn_masks, dtype=torch.int64, device=self.device)
            # pairs_batch_segment = torch.as_tensor(type_ids, dtype=torch.int64, device=self.device)

            txt_seqs = self.l_backbone.model(
                input_ids=pairs_batch_input,
                attention_mask=pairs_batch_mask,
                # token_type_ids=pairs_batch_segment
            )[0]
            txt_pad_masks = pairs_batch_mask.to(torch.bool)   # 0(False) for pad

        else:
            # for a single sentence, <s> and </s> are automatically added when calling roberta
            txt_seqs, token_inputs = self.l_backbone(txts, device=self.device)
            txt_pad_masks = token_inputs['attention_mask'].to(torch.bool)   # 0(False) for pad
        
        # txt_seqs: [B, T, D]
        # txt_pad_masks: [B, T]
        return txt_seqs, txt_pad_masks
    
    def train(self):
        for module in self.children():
            module.train()

        self.l_backbone.eval()
        if self.v_backbone.fix:
            self.v_backbone.eval()
        
        return self
    
    def eval(self):
        for module in self.children():
            module.eval()
