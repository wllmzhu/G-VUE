import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from .loss import build_loss


class CLIPDualModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.task.key == 'vl_retrieval', "CLIP-dual model is only for retrieval task"
        self.cfg = cfg

        if cfg.v_backbone.key == 'ResNet_CLIP':
            self.backbone = clip.load('RN50')[0]
        elif cfg.v_backbone.key == 'ViT_CLIP_32':
            self.backbone = clip.load('ViT-B/32')[0]
        elif cfg.v_backbone.key == 'ViT_CLIP_16':
            self.backbone = clip.load('ViT-B/16')[0]
        else:
            raise NotImplementedError

        self.v_proj = nn.Linear(self.backbone.visual.output_dim, cfg.task.decoder.embed_dim)
        self.l_proj = nn.Linear(self.backbone.visual.output_dim, cfg.task.decoder.embed_dim)

        self.initialize()

        self.criterion = build_loss(cfg.task.loss)
    
    def initialize(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if not self.cfg.v_backbone.fix:
            for p in self.backbone.visual.parameters():
                p.requires_grad_(True)

    def forward(self, imgs, txts=None, task=None):
        self.device = next(self.parameters()).device
        imgs = imgs.to(self.device)

        img_feats = self.backbone.encode_image(imgs)
        # [B, C]

        if isinstance(txts[0], (list, tuple)):
            # inference
            txt_feats = self.encode_txt(txts, expand_batch=True)
        else:
            # training
            txt_feats = self.encode_txt(txts, expand_batch=False)
        #  [B, D]

        img_feats = self.v_proj(img_feats.type(next(self.v_proj.parameters()).dtype))
        # [B, embed_dim]

        txt_feats = self.l_proj(txt_feats.type(next(self.l_proj.parameters()).dtype))
        # [B, embed_dim]

        # normalize
        img_feats = F.normalize(img_feats)
        txt_feats = F.normalize(txt_feats)

        scores = torch.matmul(img_feats, txt_feats.T)
        # [B, B] if training, [B, text_batch_size] if inference
        return scores

    @torch.no_grad()
    def encode_txt(self, txts, expand_batch=False):
        if expand_batch:
            txts = torch.cat([clip.tokenize(txt, truncate=True) for txt in txts])
        else:
            txts = clip.tokenize(txts, truncate=True)
        
        return self.backbone.encode_text(txts.to(self.device))   # [B, D]

    def train(self):
        for module in self.children():
            module.train()

        self.backbone.transformer.eval()
        if self.cfg.v_backbone.fix:
            self.backbone.visual.eval()
        
        return self
    
    def eval(self):
        for module in self.children():
            module.eval()
