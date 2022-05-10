import torch
import torch.nn as nn
from fvcore.common.registry import Registry
LOSS = Registry('Loss')


@LOSS.register()
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()


@LOSS.register()
class MSELoss(nn.MSELoss):
    def __init__(self):
        super().__init__()


@LOSS.register()
class BboxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
    
    def forward(self, pred, gt):
        return GIoU_loss(pred, gt) + self.smooth_l1_loss(pred, gt)


def GIoU_loss(pred, gt, reduction='mean'):
    # normalized in [0, 1]
    pred = pred.view(-1, 4)
    gt = gt.view(-1, 4)

    gt_area = (gt[:, 2]-gt[:, 0]) * (gt[:, 3]-gt[:, 1])
    pred_area = (pred[:, 2]-pred[:, 0]) * (pred[:, 3]-pred[:, 1])

    lt = torch.max(gt[:, :2], pred[:, :2])
    rb = torch.min(gt[:, 2:], pred[:, 2:])

    # IoU
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = gt_area + pred_area - inter
    iou = inter / union
    
    # enclosure
    lt = torch.min(gt[:, :2], pred[:, :2])
    rb = torch.max(gt[:, 2:], pred[:, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosure = wh[:, 0] * wh[:, 1]

    giou = iou - (enclosure-union)/enclosure
    loss = 1.0 - giou

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    return loss


def build_loss(cfg):
    assert cfg.key in ['CrossEntropyLoss', 'MSELoss', 'BboxLoss']
    return LOSS.get(cfg.key)
