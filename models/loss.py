import torch
import torch.nn.functional as F
from fvcore.common.registry import Registry
LOSS = Registry('Loss')


@LOSS.register()
def CrossEntropyLoss(outputs, gts):
    gts = gts.to(outputs.device)
    return F.cross_entropy(outputs, gts)


@LOSS.register()
def CrossEntropyLossWithIgnore(outputs, gts):
    gts = gts.to(outputs.device)
    return F.cross_entropy(outputs, gts, ignore_index=0)


@LOSS.register()
def VLMatchingLoss(outputs, gts):
    outputs = outputs.view(gts.shape[0], -1)
    gts = gts.to(outputs.device)
    return F.cross_entropy(outputs, gts)


@LOSS.register()
def MSELoss(outputs, gts):
    gts = gts.to(outputs.device)
    return F.mse_loss(outputs, gts)


@LOSS.register()
def DepthLoss(outputs, gts):
    min_depth, max_depth, var_val = 1e-3, 10, 0.85
    preds = outputs.sigmoid().squeeze() * max_depth
    gts = gts.squeeze().to(preds.device)
    valid_mask = torch.logical_and(gts>min_depth, gts<max_depth)
    d = torch.log(preds[valid_mask]/gts[valid_mask])
    return torch.sqrt((d ** 2).mean() - var_val * (d.mean() ** 2)) * 10.0


@LOSS.register()
def BboxLoss(outputs, gts):
    outputs = outputs.sigmoid()
    preds = torch.stack([
        outputs[:, 0]-outputs[:, 2]/2,
        outputs[:, 1]-outputs[:, 3]/2,
        outputs[:, 0]+outputs[:, 2]/2,
        outputs[:, 1]+outputs[:, 3]/2
    ], dim=1)
    gts = gts.to(preds.device)
    return 2 * GIoU_loss(preds, gts) + 5 * F.smooth_l1_loss(preds, gts)


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


def build_loss(loss_type):
    return LOSS.get(loss_type)
