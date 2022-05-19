import torch
import numpy as np
from tqdm import tqdm
from fvcore.common.registry import Registry
METRICS = Registry('Metrics')


@METRICS.register()
@torch.no_grad()
def EvalDepth(model, dataloader, cfg):
    model.eval()
    min_depth, max_depth = 1e-3, 10
    keys = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rms', 'log_rms', 'log10']
    errors_all = np.zeros((0, 8))
    total = 0
    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)

        outputs = model(imgs, txts=None)
        preds = (outputs.sigmoid().squeeze() * max_depth).clip(min_depth, max_depth)
        targets = targets.squeeze().to(preds.device)

        valid_mask = torch.logical_and(targets>min_depth, targets<max_depth)
        errors = compute_depth_errors(preds[valid_mask], targets[valid_mask])
        errors_all = np.concatenate([errors_all, errors.reshape(-1, 8)], axis=0)

        total += B
        if total >= cfg.eval.num_val_samples:
            break
    
    errors_all = errors_all.mean(0)
    error_dict = {}
    for i, k in enumerate(keys):
        error_dict.update({k: errors_all[i]})
    return error_dict


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
        preds = torch.topk(outputs, k=1, dim=-1).indices.squeeze()

        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)
        targets = targets.to(preds.device)

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
        targets = targets.to(preds.device)
        iou = box_iou(preds, targets)
        
        acc += (iou >= 0.5).sum()
        total += B
        if total >= cfg.eval.num_val_samples:
            break

    return {'Bbox Acc@0.5': (acc/total).item()}


def box_iou(a, b):
    a = a.view(-1, 4)
    b = b.view(-1, 4)

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


def compute_depth_errors(pred, gt):
    delta = torch.maximum((pred / gt), (gt / pred))
    d1 = (delta < 1.25).float().mean()
    d2 = (delta < 1.25 ** 2).float().mean()
    d3 = (delta < 1.25 ** 3).float().mean()

    rms = (pred - gt) ** 2
    rms = torch.sqrt(rms.mean())

    log_rms = (torch.log(pred/gt)) ** 2
    log_rms = torch.sqrt(log_rms.mean())

    abs_rel = ((pred - gt).abs() / gt).mean()
    sq_rel = ((pred - gt) ** 2 / gt).mean()

    log10 = torch.log10(pred/gt).abs().mean()

    errors = [d1, d2, d3, abs_rel, sq_rel, rms, log_rms, log10]
    return np.array([v.detach().cpu().numpy() for v in errors])


def build_evaluator(eval_type):
    return METRICS.get(eval_type)
