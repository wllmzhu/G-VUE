import torch
import torch.nn.functional as F
import numpy as np
import h5py
from math import ceil
from tqdm import tqdm
import ast


def EvalDepth(dataloader, h5py_file):
    grp = h5py_file['depth']
    subset = dataloader.dataset.subset
    all_preds = np.array(grp[subset])
    print(f'evaluating depth task on NYUv2 {subset} subset')

    min_depth, max_depth = 1e-3, 10
    keys = ['d1', 'abs_rel', 'rms']
    errors_all = []
    sample_idx = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)
        if not isinstance(targets, torch.Tensor):
            targets = torch.stack(targets)
        targets = targets.squeeze().detach().cpu().numpy()

        for i in range(B):
            target = targets[i]
            valid_mask = np.logical_and(target>min_depth, target<max_depth)
            errors = compute_depth_errors(all_preds[sample_idx][valid_mask], target[valid_mask])
            errors_all.append(errors)
            sample_idx += 1
    
    errors_all = np.concatenate(errors_all, axis=0)
    errors_all = errors_all.mean(0)
    error_dict = {}
    for i, k in enumerate(keys):
        error_dict.update({k: round(errors_all[i], 4)})
    print(error_dict)

    err2per = np.exp(-1.386*errors_all[1:])
    task_score = (errors_all[0]+err2per.sum()) / 3 * 100
    return task_score


def EvalCameraRelocalization(dataloader, h5py_file):
    grp = h5py_file['camera_relocalization']
    subset = dataloader.dataset.scene
    all_preds = np.array(grp[subset])
    print(f'evaluating camera_relocalization task on {dataloader.dataset.info.dataset} {subset} subset')

    position_error = []
    orientation_error = []
    total = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = targets.shape[0]
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        else:
            targets = np.array(targets)
        
        pos_error, orient_error = camera_relocalization_errors(all_preds[total: total+B], targets)
        position_error.append(pos_error)
        orientation_error.append(orient_error)

        total += B

    position_error = np.concatenate(position_error, axis=0)
    orientation_error = np.concatenate(orientation_error, axis=0)
    err_trans = np.mean(position_error)
    err_orient = np.mean(orientation_error)
    print(f'T: median = {np.median(position_error):.3f}, mean = {err_trans:.3f}')
    print(f'R: median = {np.median(orientation_error):.3f}, mean = {err_orient:.3f}')

    task_score = (np.exp(-1.386*err_trans) + np.exp(-1.386*err_orient)) / 2 * 100
    return task_score


def EvalRec(dataloader, h5py_file):
    grp = h5py_file['camera_relocalization']
    subset = dataloader.dataset.subset
    all_preds = np.array(grp[subset])
    print(f'evaluating 3d_reconstruction task on ShapeNet {subset} subset')
    iou = 0
    total = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        tsdf = torch.stack([ele[1] for ele in targets])
        B = tsdf.shape[0]
        pred = all_preds[total: total+B]
        vox = (tsdf < 0).detach().cpu().numpy()

        area_intersect = np.logical_and(pred, vox).reshape(B, -1)
        area_union = np.logical_or(pred, vox).reshape(B, -1)
        iou += (area_intersect.sum(-1) / area_union.sum(-1)).sum()
        total += B

    task_score = iou / total * 100
    print(f'Rec IoU: {round(task_score, 2)}')
    return task_score


def EvalRetrieval(dataloader, h5py_file):
    grp = h5py_file['vl_retrieval']
    subset = dataloader.dataset.subset
    all_preds = np.array(grp[subset])
    print(f'evaluating vl_retrieval task on Flickr30k {subset} subset')
    img2txt_id = dataloader.dataset.img2txt
    highest_ranks = []
    total = 0

    for data in tqdm(dataloader):
        imgs, _, targets = data
        B = len(targets)
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        for i in range(B):
            txt_gt_ids = img2txt_id[targets[i]]   # 5 ground truth candidates
            # get highest rank among 5 candidates
            for idx, txt_id in enumerate(all_preds[total+i]):
                if txt_id in txt_gt_ids:
                    break
            highest_ranks.append(idx)
    
        total += B
    
    highest_ranks = np.array(highest_ranks)
    recall_1 = (highest_ranks < 1).sum() / total * 100
    recall_5 = (highest_ranks < 5).sum() / total * 100
    recall_10 = (highest_ranks < 10).sum() / total * 100
    print(f'Recall@1: {round(recall_1, 2)}, Recall@5: {round(recall_5, 2)}, Recall@10: {round(recall_10, 2)}')
    task_score = (recall_1+recall_5+recall_10) / 3
    return task_score


def EvalBbox(dataloader, h5py_file):
    grp = h5py_file['phrase_grounding']
    subset = dataloader.dataset.subset
    all_preds = np.array(grp[subset])
    print(f'evaluating phrase_grounding task on RefCOCO {subset} subset')
    acc = 0 
    total = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        else:
            targets = np.array(targets)
        
        iou = box_iou(all_preds[total: total+B], targets)
        acc += (iou >= 0.5).sum()
        total += B
    
    task_score = acc / total * 100
    print(f'Bbox Acc@0.5: {round(task_score, 2)}')
    return task_score


def EvalSeg(dataloader, h5py_file):
    grp = h5py_file['segmentation']
    subset = dataloader.dataset.subset
    all_preds = np.array(grp[subset])
    print(f'evaluating segmentation task on ADE20k {subset} subset')

    num_classes = 151
    total = 0
    # 151 - 1 = 150
    total_area_intersect = np.zeros((num_classes-1, ), dtype=np.float)
    total_area_union = np.zeros((num_classes-1, ), dtype=np.float)

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)

        if not isinstance(targets, torch.Tensor):
            targets = torch.stack(targets)
        targets = targets.detach().cpu().numpy()

        area_intersect, area_union = intersect_and_union(all_preds[total: total+B], targets, num_classes, ignore_index=0)[:2]
        total_area_intersect += area_intersect
        total_area_union += area_union

        total += B
    
    iou = total_area_intersect / total_area_union * 100
    task_score = np.nanmean(iou)
    print(f'mIoU: {round(task_score, 2)}')
    return task_score


def EvalQA(dataloader, h5py_file):
    grp = h5py_file['vqa']
    subset = dataloader.dataset.subset
    all_preds = np.array(grp[subset])
    print(f'evaluating vqa task on GQA {subset} subset')
    acc = 0
    total = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        else:
            targets = np.array(targets)

        acc += (all_preds[total: total+B]==targets).sum()
        total += B

    task_score = acc / total * 100
    print(f'Answer Acc: {round(task_score, 2)}')
    return task_score


def EvalVCR(dataloader, h5py_file):
    grp = h5py_file['common_sense']
    subset = dataloader.dataset.subset
    all_preds = np.array(grp[subset])
    print(f'evaluating common_sense task on VCR {subset} subset')
    acc = 0
    total = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        else:
            targets = np.array(targets)

        acc += (all_preds[total: total+B]==targets).sum()
        total += B

    task_score = acc / total * 100
    print(f'Answer Acc: {round(task_score, 2)}')
    return task_score


def EvalBongard(dataloader, h5py_file):
    grp = h5py_file['bongard']
    subset = dataloader.dataset.subset
    all_preds = np.array(grp[subset])
    print(f'evaluating bongard task on Bongard-HOI {subset} subset')
    acc = 0
    total = 0

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        else:
            targets = np.array(targets)

        targets = targets.flatten()   # [B, 2] -> [2B]
        acc += (all_preds[total: total+2*B]==targets).sum()
        total += 2*B

    task_score = acc / total * 100
    print(f'Answer Acc: {round(task_score, 2)}')
    return task_score


def EvalNav(env_name, evaluator, h5py_file):
    task_record = h5py_file['navigation'][env_name]
    task_record = ast.literal_eval(np.array2string(np.array(task_record))[2:-1]) 
    print(f'evaluating navigation task on R2R {env_name} subset')
    score_summary, _ = evaluator.score(task_record)
    #acc_and_spl = [score_summary['success_rate'], score_summary['spl']]
    task_score = score_summary['spl'] * 100
    return task_score
    

def EvalManip(eval_task, h5py_file):
    grp = h5py_file['manipulation']
    task_record = dict(grp[eval_task])
    print(f'evaluating manipulation task on Ravens {eval_task} subset')
    task_score = np.mean(task_record['rewards']) * 100
    print(f'Success Score: {task_score:.2f}')
    return task_score


def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, ), normalized 
    :param q2: (4, ), normalized
    :return:
    """
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta

def camera_relocalization_errors(output, target):
    # returns mean position error in units and mean angle error in degrees
    output = torch.from_numpy(output)
    target = torch.from_numpy(target)

    position = output[:, :3]
    orientation = output[:, -4:]
    position_target = target[:, :3]
    orientation_target = target[:, -4:]
    orientation = F.normalize(orientation, p=2, dim=1)
    orientation_target = F.normalize(orientation_target, p=2, dim=1)

    position_error = torch.norm(position - position_target, p=2, dim=-1).cpu().numpy()
    
    orientation_error = [quaternion_angular_error(p, t) for p, t in zip(orientation.cpu().numpy(), orientation_target.cpu().numpy())]
    orientation_error = np.array(orientation_error)

    return position_error, orientation_error

def box_iou(a, b):
    """ here a and b are both numpy.ndarray """
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    a_area = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    b_area = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])

    lt = np.maximum(a[:, :2], b[:, :2])
    rb = np.minimum(a[:, 2:], b[:, 2:])

    # IoU
    wh = np.clip(rb-lt, a_min=0, a_max=None)
    inter = wh[:, 0] * wh[:, 1]
    union = a_area + b_area - inter
    iou = inter / union
    # [B]
    return iou


def compute_depth_errors(pred, gt):
    """ here pred and gt are both numpy.ndarray """
    delta = np.maximum((pred / gt), (gt / pred))
    d1 = (delta < 1.25).astype(np.float).mean()

    abs_rel = (np.abs(pred - gt) / gt).mean()

    rms = (pred - gt) ** 2
    rms = np.sqrt(rms.mean())

    return np.array([d1, abs_rel, rms])


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


task_eval_dict = {
    'depth': EvalDepth, 'camera_relocalization': EvalCameraRelocalization, '3d_reconstruction': EvalRec,
    'vl_retrieval': EvalRetrieval, 'phrase_grounding': EvalBbox, 'segmentation': EvalSeg,
    'vqa': EvalQA, 'common_sense': EvalVCR, 'bongard': EvalBongard,
    'navigation': EvalNav, 'manipulation': EvalManip
}


def evaluate(task):
    return task_eval_dict[task]
