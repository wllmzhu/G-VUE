import torch
import timm
import clip
import torch.nn as nn
from fvcore.common.registry import Registry
BACKBONE = Registry('Backbone')


@BACKBONE.register()
class ResNet_ImageNet(nn.Module):
    """
    return five 
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True)
    
    @torch.no_grad()
    def forward(self, imgs):
        return self.backbone(imgs)


@BACKBONE.register()
class ResNet_Ego4D(nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def forward(self, imgs):
        pass


@BACKBONE.register()
class ViT_CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.preprocess = clip.load('ViT-B/32')
    
    @torch.no_grad()
    def forward(self, imgs):
        imgs = self.preprocess(imgs)
        return self.backbone(imgs)


@BACKBONE.register()
class Customized(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, imgs):
        pass


def build_v_backbone(backbone_key):
    assert backbone_key in [
        'ResNet_ImageNet',
        'ResNet_Ego4D',
        'ViT_CLIP',
        'Customized'
    ]
    return BACKBONE.get(backbone_key)


# def inference(imgs):
#     res_input = []
#     vit_input = []
#     for img in imgs:
#         res_input.append(preprocess_clip_res(img))
#         vit_input.append(preprocess_clip_vit(img))
#     res_input = torch.stack(res_input).to('cuda')
#     vit_input = torch.stack(vit_input).to('cuda')
#     # txt_input = clip.tokenize([f'a photo of {txt}']).to('cuda')

#     with torch.no_grad():
#         # features in V-L aligned space
#         vl_res_feature_vl = clip_res.encode_image(res_input)   # [3, 1024]
#         # vl_res_feature_l = clip_res.encode_text(txt_input)   # [1, 1024]
#         vl_vit_feature_vl = clip_vit.encode_image(vit_input)   # [3, 512]
#         # vl_vit_feature_l = clip_vit.encode_text(txt_input)   # [1, 512]

#         # features in vision space
#         v_res_feature_v = v_res(res_input)   # [3, 2048]
#         v_vit_feature_v = v_vit(vit_input)   # [3, 768]
#         vl_res_feature_v = clip_res_v.encode_image(res_input).flatten(1)   # [3, 2048]
#         vl_vit_feature_v = clip_vit_v.encode_image(vit_input)   # [3, 768]

#     return {
#         'v_res_feature_v': v_res_feature_v,
#         'v_vit_feature_v': v_vit_feature_v,
#         'vl_res_feature_v': vl_res_feature_v,
#         'vl_vit_feature_v': vl_vit_feature_v,
#         'vl_res_feature_vl': vl_res_feature_vl,
#         'vl_vit_feature_vl': vl_vit_feature_vl
#     }
