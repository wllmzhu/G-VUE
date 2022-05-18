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

        self.extract_fs = []
        self._register_hooks(cfg.extract_layer)

        self.eval()
    
    def _register_hooks(self, layers):
        for block_idx, block in enumerate(self.backbone.transformer.resblocks):
            if block_idx in layers:
                block.register_forward_hook(self._feature_hook())
    
    def _feature_hook(self):
        def _hook(model, input, output):
            self.extract_fs.append(output)
        return _hook
    
    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype

    @torch.no_grad()
    def forward(self, imgs):
        B, h, w = imgs.shape[0], imgs.shape[-2]//32, imgs.shape[-1]//32
        imgs_ori_type = imgs.dtype
        imgs = imgs.type(self.dtype)   # HalfTensor

        self.extract_fs = []
        _ = self.backbone(imgs)

        imgs = imgs.type(imgs_ori_type)
        # [1+hw, B, C] -> [B, C, h, w]
        return [f[1:, ...].permute(1, 2, 0).view(B, -1, h, w).type(imgs_ori_type) 
                for f in self.extract_fs]


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
