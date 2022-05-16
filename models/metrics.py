import torch
import numpy as np
from tqdm import tqdm
from fvcore.common.registry import Registry
METRICS = Registry('Metrics')


@METRICS.register()
@torch.no_grad()
def EvalQA(model, dataloader, cfg):
    model.eval()
    acc = 0
    total = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)

        outputs = model(imgs, txts)
        _, preds = outputs.topk(k=1, dim=1)

        acc += (preds==targets).sum()
        total += B
        if total >= cfg.eval.num_val_samples:
            break

    return {'Answer Acc': (acc/total).item()}


@METRICS.register()
@torch.no_grad()
def EvalBbox(model, dataloader, cfg):
    model.eval()
    acc = 0 
    total = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)

        outputs = model(imgs, txts).sigmoid()
        preds = torch.stack([
            outputs[:, 0]-outputs[:, 2]/2,
            outputs[:, 1]-outputs[:, 3]/2,
            outputs[:, 0]+outputs[:, 2]/2,
            outputs[:, 1]+outputs[:, 3]/2
        ], dim=1)
        iou = box_iou(preds, targets)
        
        acc += (iou >= 0.5).sum()
        total += B
        if total >= cfg.eval.num_val_samples:
            break

    return {'Bbox Acc@0.5': (acc/total).item()}


def box_iou(a, b):
    a = a.view(-1, 4)
    b = b.view(-1, 4).to(a.device)

    a_area = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    b_area = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])

    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])

    # IoU
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = a_area + b_area - inter
    iou = inter / union
    # [B]
    return iou


def build_evaluator(cfg):
    assert cfg.key in ['EvalBbox']
    return METRICS.get(cfg.key)
