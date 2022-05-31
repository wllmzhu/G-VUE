import torch
import numpy as np
from tqdm import tqdm
from fvcore.common.registry import Registry
METRICS = Registry('Metrics')


@METRICS.register()
@torch.no_grad()
def EvalBongard(model, dataloader, cfg):
    model.eval()
    acc = 0
    total = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)

        outputs = model(imgs, txts, cfg.task.key)
        # [2B, 2] -> [2B]
        preds = torch.topk(outputs, k=1, dim=-1).indices.squeeze()

        targets = targets.flatten().to(preds.device)   # [B, 2] -> [2B]

        acc += (preds==targets).sum()
        total += 2*B
        if total >= cfg.eval.num_val_samples:
            break

    return {'Answer Acc': (acc/total).item()}


@METRICS.register()
@torch.no_grad()
def EvalRetrieval(model, dataloader, cfg):
    # image-to-text retrieval
    model.eval()
    total = 0
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)

        outputs = model(imgs, txts, cfg.task.key)
        outputs = outputs.view(B, -1)

        top1_preds = torch.topk(outputs, k=1, dim=-1).indices
        top5_preds = torch.topk(outputs, k=5, dim=-1).indices
        top10_preds = torch.topk(outputs, k=10, dim=-1).indices

        for i in range(B):
            if targets[i] in top1_preds[i]:
                recall_1 += 1
            if targets[i] in top5_preds[i]:
                recall_5 += 1
            if targets[i] in top10_preds[i]:
                recall_10 += 1

        total += B
        if total >= cfg.eval.num_val_samples:
            break

    return {'Recall@1': recall_1/total, 'Recall@5': recall_5/total, 'Recall@10': recall_10/total}


@METRICS.register()
@torch.no_grad()
def EvalVCR(model, dataloader, cfg):
    model.eval()
    acc = 0
    total = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)

        outputs = model(imgs, txts, cfg.task.key)
        # [4B, 1] -> [B, 4]
        outputs = outputs.view(B, -1)
        preds = torch.topk(outputs, k=1, dim=-1).indices.squeeze()   # [B]

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
def EvalSeg(model, dataloader, cfg):
    model.eval()
    num_classes = cfg.task.num_classes
    total = 0
    # 151 - 1 = 150
    total_area_intersect = np.zeros((num_classes-1, ), dtype=np.float)
    total_area_union = np.zeros((num_classes-1, ), dtype=np.float)

    for data in dataloader:
        imgs, txts, targets = data
        B = len(targets)

        outputs = model(imgs, txts=None)
        preds = outputs.argmax(1).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        area_intersect, area_union = intersect_and_union(preds, targets, num_classes, ignore_index=0)[:2]
        total_area_intersect += area_intersect
        total_area_union += area_union

        total += B
        if total >= cfg.eval.num_val_samples:
            break
    
    iou = total_area_intersect / total_area_union
    return {'mIoU': np.nanmean(iou)}


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

@METRICS.register()
@torch.no_grad()
def EvalRec(model, dataloader, cfg):
    model.eval()
    iou = 0
    total = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        idx, tsdf = targets
        B = tsdf.shape[0]

        img_logits = model(imgs)
        outputs = model.decoder.inference(img_logits).squeeze(1)
        pred_vox = outputs < 0
        vox = tsdf < 0
        area_intersect = torch.logical_and(pred_vox, vox).reshape(B, -1)
        area_union = torch.logical_or(pred_vox, vox).reshape(B, -1)
        iou += (area_intersect / area_union).mean()
        total += 1

    return {'Rec iou': (iou/total).item()}

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


def intersect_and_union(pred_label, label, num_classes, ignore_index):
    """Calculate intersection and Union.
    Args:
        pred_label (ndarray): Prediction segmentation map
        label (ndarray): Ground truth segmentation map
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes
         ndarray: The union of prediction and ground truth histogram on all
             classes
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes)+1)
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes)+1)
    area_label, _ = np.histogram(label, bins=np.arange(num_classes)+1)
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def build_evaluator(eval_type):
    return METRICS.get(eval_type)
