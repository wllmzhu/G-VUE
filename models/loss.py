import torch
import torch.nn as nn
from fvcore.common.registry import Registry
LOSS = Registry('Loss')


@LOSS.register()
class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
