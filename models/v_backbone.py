import torch
import timm
import clip
import torch.nn as nn
from .decoder_utils import resize
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
        self.fix = cfg.fix
        if self.fix:
            self.backbone.eval()
    
    def forward(self, imgs):
        if self.fix:
            with torch.no_grad():
                return self.backbone(imgs)
        else:
            return self.backbone(imgs)


@BACKBONE.register()
class ResNet_MoCov3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=False, features_only=True)
        curr_sd = self.backbone.state_dict()
        load_sd = torch.load(cfg.ckpt_path)['state_dict']
        for k in load_sd.keys():
            if 'base_encoder' in k:
                curr_k = k.replace('module.base_encoder.', '')
                if curr_k in curr_sd:
                    curr_sd[curr_k] = load_sd[k]

        self.backbone.load_state_dict(curr_sd)

        self.fix = cfg.fix
        if self.fix:
            self.backbone.eval()
    
    def forward(self, imgs):
        if self.fix:
            with torch.no_grad():
                return self.backbone(imgs)
        else:
            return self.backbone(imgs)


@BACKBONE.register()
class ResNet_Ego4D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=False, features_only=True)
        ego4d_weights = load_r3m('resnet50').module.convnet.state_dict()
        self.backbone.load_state_dict(ego4d_weights)

        self.fix = cfg.fix
        if self.fix:
            self.backbone.eval()
    
    def forward(self, imgs):
        if self.fix:
            with torch.no_grad():
                return self.backbone(imgs)
        else:
            return self.backbone(imgs)


@BACKBONE.register()
class ResNet_CLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = clip.load('RN50')[0].visual

        self.extract_fs = []
        self._register_hooks()

        self.fix = cfg.fix
        if self.fix:
            self.backbone.eval()
    
    def _register_hooks(self):
        self.backbone.layer1.register_forward_hook(self._feature_hook())
        self.backbone.layer2.register_forward_hook(self._feature_hook())
        self.backbone.layer3.register_forward_hook(self._feature_hook())
        self.backbone.layer4.register_forward_hook(self._feature_hook())
    
    def _feature_hook(self):
        def _hook(model, input, output):
            self.extract_fs.append(output)
        return _hook
    
    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype

    def extract_features(self, imgs):
        imgs_ori_type = imgs.dtype
        imgs = imgs.type(self.dtype)   # HalfTensor

        self.extract_fs = []
        _ = self.backbone(imgs)

        imgs = imgs.type(imgs_ori_type)
        return [f.type(imgs_ori_type) for f in self.extract_fs]
    
    def forward(self, imgs):
        if self.fix:
            with torch.no_grad():
                return self.extract_features(imgs)
        else:
            return self.extract_features(imgs)


@BACKBONE.register()
class ViT_CLIP_32(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = clip.load('ViT-B/32')[0].visual
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.reductions = cfg.reduction

        self.extract_fs = []
        self._register_hooks(cfg.extract_layer)

        self.fix = cfg.fix
        if self.fix:
            self.backbone.eval()
    
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

    def extract_features(self, imgs):
        B, h, w = imgs.shape[0], imgs.shape[-2]//self.patch_size, imgs.shape[-1]//self.patch_size
        imgs_ori_type = imgs.dtype
        imgs = imgs.type(self.dtype)   # HalfTensor

        self.extract_fs = []
        _ = self.backbone(imgs)

        # [1+hw, B, C] -> [B, C, h, w]
        fs = [f[1:, ...].permute(1, 2, 0).view(B, -1, h, w).type(imgs_ori_type) 
              for f in self.extract_fs]
        # create different resolutions
        for i in range(len(self.reductions)):
            if self.reductions[i] != self.patch_size:
                fs[i] = resize(fs[i], size=self.image_size//self.reductions[i], mode='bilinear')
        
        imgs = imgs.type(imgs_ori_type)
        return fs
    
    def forward(self, imgs):
        if self.fix:
            with torch.no_grad():
                return self.extract_features(imgs)
        else:
            return self.extract_features(imgs)


@BACKBONE.register()
class ViT_CLIP_16(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = clip.load('ViT-B/16')[0].visual
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.reductions = cfg.reduction

        self.extract_fs = []
        self._register_hooks(cfg.extract_layer)

        self.fix = cfg.fix
        if self.fix:
            self.backbone.eval()
    
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

    def extract_features(self, imgs):
        B, h, w = imgs.shape[0], imgs.shape[-2]//self.patch_size, imgs.shape[-1]//self.patch_size
        imgs_ori_type = imgs.dtype
        imgs = imgs.type(self.dtype)   # HalfTensor

        self.extract_fs = []
        _ = self.backbone(imgs)

        # [1+hw, B, C] -> [B, C, h, w]
        fs = [f[1:, ...].permute(1, 2, 0).view(B, -1, h, w).type(imgs_ori_type) 
              for f in self.extract_fs]
        # create different resolutions
        for i in range(len(self.reductions)):
            if self.reductions[i] != self.patch_size:
                fs[i] = resize(fs[i], size=self.image_size//self.reductions[i], mode='bilinear')
        
        imgs = imgs.type(imgs_ori_type)
        return fs
    
    def forward(self, imgs):
        if self.fix:
            with torch.no_grad():
                return self.extract_features(imgs)
        else:
            return self.extract_features(imgs)


@BACKBONE.register()
class ViT_MAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.image_size
        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0, img_size=self.image_size
        )
        self.patch_size = cfg.patch_size
        self.reductions = cfg.reduction

        load_sd = torch.load(cfg.ckpt_path)['model']
        curr_sd = self.backbone.state_dict()
        for k in load_sd.keys():
            if k in curr_sd and curr_sd[k].shape == load_sd[k].shape:
                curr_sd[k] = load_sd[k]
        self.backbone.load_state_dict(curr_sd)
        # self.backbone.load_state_dict(load_sd)

        self.extract_fs = []
        self._register_hooks(cfg.extract_layer)

        self.fix = cfg.fix
        if self.fix:
            self.backbone.eval()
    
    def _register_hooks(self, layers):
        for block_idx, block in enumerate(self.backbone.blocks):
            if block_idx in layers:
                block.register_forward_hook(self._feature_hook())
    
    def _feature_hook(self):
        def _hook(model, input, output):
            self.extract_fs.append(output)
        return _hook

    def extract_features(self, imgs):
        B, h, w = imgs.shape[0], imgs.shape[-2]//self.patch_size, imgs.shape[-1]//self.patch_size

        self.extract_fs = []
        _ = self.backbone(imgs)

        # [B, 1+hw, C] -> [B, C, h, w]
        fs = [f[:, 1:, :].permute(0, 2, 1).view(B, -1, h, w) for f in self.extract_fs]
        # create different resolutions
        for i in range(len(self.reductions)):
            if self.reductions[i] != self.patch_size:
                fs[i] = resize(fs[i], size=self.image_size//self.reductions[i], mode='bilinear')
        
        return fs
    
    def forward(self, imgs):
        if self.fix:
            with torch.no_grad():
                return self.extract_features(imgs)
        else:
            return self.extract_features(imgs)


@BACKBONE.register()
class Customized(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pass
    
    def forward(self, imgs):
        # return List([B, C, h, w])
        pass


def build_v_backbone(cfg):
    print(f'initializing {cfg.key} visual backbone')
    return BACKBONE.get(cfg.key)(cfg)
