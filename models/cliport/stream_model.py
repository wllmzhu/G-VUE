import torch
import torch.nn as nn
import torch.nn.functional as F
from models.v_backbone import build_v_backbone
from models.l_backbone import RoBERTa
from fvcore.common.registry import Registry
from .utils import FusionMult, Up, IdentityBlock, ConvBlock


STREAM = Registry('Stream')


@STREAM.register()
class ResNetStream(nn.Module):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.input_dim = cfg.backbone.hidden_dim[-1]
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']
        self.lang_fusion_type = self.cfg['train']['lang_fusion_type']
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.preprocess = preprocess

        self.v_backbone = build_v_backbone(cfg.backbone)
        self.l_backbone = RoBERTa(cache_dir=cfg.l_backbone.cfg_dir)
        # self._load_clip()
        self._build_decoder()

        self.initialize()
    
    def initialize(self):
        for p in self.l_backbone.parameters():
            p.requires_grad_(False)
        if self.v_backbone.fix:
            for p in self.v_backbone.parameters():
                p.requires_grad_(False)

    def _build_decoder(self):
        # language
        self.lang_fuser1 = FusionMult(input_dim=self.input_dim // 2)
        self.lang_fuser2 = FusionMult(input_dim=self.input_dim // 4)
        self.lang_fuser3 = FusionMult(input_dim=self.input_dim // 8)

        self.proj_input_dim = self.cfg.l_backbone.hidden_dim
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.up1 = Up(2048, 1024 // self.up_factor, self.bilinear)

        self.up2 = Up(1024, 512 // self.up_factor, self.bilinear)

        self.up3 = Up(512, 256 // self.up_factor, self.bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

    def forward(self, x, l):
        x = self.preprocess(x)

        in_type = x.dtype
        in_shape = x.shape
        x = x[:, :3]   # select RGB
        v_feature_list = self.v_backbone(x)
        x = v_feature_list[-1]
        x = x.to(in_type)

        txt_seqs, _ = self.l_backbone(l, device=x.device)
        l_input = txt_seqs[:, 0].to(dtype=x.dtype)

        assert x.shape[1] == self.input_dim
        x = self.conv1(x)

        x = self.lang_fuser1(x, l_input, x2_proj=self.lang_proj1)
        x = self.up1(x, v_feature_list[-2])

        x = self.lang_fuser2(x, l_input, x2_proj=self.lang_proj2)
        x = self.up2(x, v_feature_list[-3])

        x = self.lang_fuser3(x, l_input, x2_proj=self.lang_proj3)
        x = self.up3(x, v_feature_list[-4])

        for layer in [self.layer1, self.layer2, self.layer3, self.conv2]:
            x = layer(x)

        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return x
