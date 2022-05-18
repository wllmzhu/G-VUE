import torch
import timm
import clip
import torch.nn as nn
from .transformer import TransformerDecoderLayer
from .positional_embedding import build_positional_embedding
from r3m import load_r3m
from fvcore.common.registry import Registry
BACKBONE = Registry('Backbone')


@BACKBONE.register()
class ResNet_ImageNet(nn.Module):
    """
    return five levels of features
    """
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True)
        self.backbone.eval()
    
    @torch.no_grad()
    def forward(self, imgs):
        return self.backbone(imgs)


@BACKBONE.register()
class ResNet_Ego4D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=False, features_only=True)
        ego4d_weights = load_r3m('resnet50').module.convnet.state_dict()
        self.backbone.load_state_dict(ego4d_weights)
        self.backbone.eval()
    
    @torch.no_grad()
    def forward(self, imgs):
        return self.backbone(imgs)


@BACKBONE.register()
class ViT_CLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = clip.load('ViT-B/32')[0].visual
        # need to register hooks

        reductions = cfg.reduction
        hidden_dims = cfg.hidden_dim
        num_extractions = len(cfg.extract_layer)

        self.patch_embeds = []
        self.cross_blocks = []
        self.pos_embeds = []
        for i in range(num_extractions):
            self.patch_embeds.append(nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=hidden_dims[i],
                        kernel_size=reductions[i], stride=reductions[i]),
                nn.LayerNorm(hidden_dims[i])
            ))

            self.cross_blocks.append(CrossBlock(
                q_dim=hidden_dims[i], kv_dim=hidden_dims[-1], nheads=cfg.nheads
            ))

            self.pos_embeds.append(build_positional_embedding(
                type=cfg.positional_embedding,
                shape=(cfg.image_size//reductions[i], cfg.image_size//reductions[i]),
                hidden_dim=hidden_dims[i]
            ))

        self.eval()
    
    @torch.no_grad()
    def forward(self, imgs):
        f, c_last = self.backbone(imgs)
        v_feature_list = []
        for i in range(len(self.patch_embeds)):
            c = self.patch_embeds[i](imgs)
            c = self.cross_blocks[i](
                q=c, kv=f[i],
                pos_embed_q=self.pos_embeds[i](imgs.shape[0]),
                pos_embed_kv=None   # kv already containing positional features
            )
            v_feature_list.append(c)
        v_feature_list.append(c_last)
        return v_feature_list


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
        q & kv: [B, C, h, w]
        pos_embed_q & pos_embed_kv: [B, hw, C]
        """
        if self.proj is not None:
            kv = self.proj(kv)
        return self.cross_attn(
            tgt=q.flatten(2).permute(2, 0, 1), memory=kv.flatten(2).permute(2, 0, 1),
            query_pos=pos_embed_q.permute(2, 0, 1) if pos_embed_q is not None else None,
            pos=pos_embed_kv.permute(2, 0, 1) if pos_embed_kv is not None else None
        )


@BACKBONE.register()
class Customized(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pass
    
    def forward(self, imgs):
        # return List([B, C, h, w])
        pass


def build_v_backbone(cfg):
    assert cfg.key in [
        'ResNet_ImageNet',
        'ResNet_Ego4D',
        'ViT_CLIP',
        'Customized'
    ]
    return BACKBONE.get(cfg.key)(cfg)

# vl_res_feature_vl = clip_res.encode_image(res_input)   # [3, 1024]
# vl_vit_feature_vl = clip_vit.encode_image(vit_input)   # [3, 512]

# # features in vision space
# vl_res_feature_v = clip_res_v.encode_image(res_input).flatten(1)   # [3, 2048]
# vl_vit_feature_v = clip_vit_v.encode_image(vit_input)   # [3, 768]
