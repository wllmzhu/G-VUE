import torch
import timm
import clip
import torch.nn as nn
from r3m import load_r3m
from fvcore.common.registry import Registry
BACKBONE = Registry('Backbone')


@BACKBONE.register()
class ResNet_ImageNet(nn.Module):
    """
    return five levels of features
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True)
        self.backbone.eval()
    
    @torch.no_grad()
    def forward(self, imgs):
        return self.backbone(imgs)


@BACKBONE.register()
class ResNet_Ego4D(nn.Module):
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.backbone = clip.load('ViT-B/32')[0]
        # TO BE IMPLEMENTED: feature extraction
        self.backbone.eval()
    
    @torch.no_grad()
    def forward(self, imgs):
        return self.backbone(imgs)


@BACKBONE.register()
class Customized(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, imgs):
        pass


def build_v_backbone(cfg):
    assert cfg.key in [
        'ResNet_ImageNet',
        'ResNet_Ego4D',
        'ViT_CLIP',
        'Customized'
    ]
    return BACKBONE.get(cfg.key)()

# vl_res_feature_vl = clip_res.encode_image(res_input)   # [3, 1024]
# vl_vit_feature_vl = clip_vit.encode_image(vit_input)   # [3, 512]

# # features in vision space
# vl_res_feature_v = clip_res_v.encode_image(res_input).flatten(1)   # [3, 2048]
# vl_vit_feature_v = clip_vit_v.encode_image(vit_input)   # [3, 768]
