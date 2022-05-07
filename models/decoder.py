import torch
import torch.nn as nn
from fvcore.common.registry import Registry
DECODER = Registry('Decoder')


@DECODER.register()
class QueryType(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img_seqs, txt_seqs=None, txt_pad_masks=None, train=True):
        pass


@DECODER.register()
class LabelType(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, joint_seqs):
        pass


@DECODER.register()
class DenseType(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, joint_seqs):
        pass


def build_decoder(decoder_type):
    assert decoder_type in ['QueryType', 'LabelType', 'DenseType']
    return DECODER.get(decoder_type)
