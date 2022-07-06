import h5py
import numpy as np


def EvalDepth(pred_h5py, gt_h5py):
    min_depth, max_depth = 1e-3, 10
    keys = ['d1', 'abs_rel', 'rms']
    errors_all = []

    all_preds = np.array(pred_h5py['depth']['test'])
    all_gts = np.array(gt_h5py['depth']['test'])
    subset_size = all_preds.shape[0]
    print(f'evaluating depth task on NYUv2 test subset, size at {subset_size}')

    for i in range(subset_size):
        target = all_gts[i]
        valid_mask = np.logical_and(target>min_depth, target<max_depth)
        errors = compute_depth_errors(all_preds[i][valid_mask], target[valid_mask])
        errors_all.append(errors.reshape(1, -1))
    
    errors_all = np.concatenate(errors_all, axis=0)
    errors_all = errors_all.mean(0)
    error_dict = {}
    for i, k in enumerate(keys):
        error_dict.update({k: round(errors_all[i], 4)})
    print(error_dict)

    err2per = np.exp(-1.386*errors_all[1:])
    task_score = (errors_all[0]+err2per.sum()) / 3 * 100
    return task_score


def EvalCameraRelocalization(pred_h5py, gt_h5py):
    scene2dataset = {
        'KingsCollege': 'CambridgeLandmarks', 'OldHospital': 'CambridgeLandmarks',
        'ShopFacade': 'CambridgeLandmarks', 'StMarysChurch': 'CambridgeLandmarks',
        'chess': '7Scenes', 'fire': '7Scenes', 'heads': '7Scenes', 'office': '7Scenes',
        'pumpkin': '7Scenes', 'redkitchen': '7Scenes', 'stairs': '7Scenes'
    }
    position_error_cl = []
    position_error_7s = []
    orientation_error_cl = []
    orientation_error_7s = []
    for subset in pred_h5py['camera_relocalization'].keys():
        all_preds = np.array(pred_h5py['camera_relocalization'][subset])
        all_gts = np.array(gt_h5py['camera_relocalization'][subset])
        subset_size = all_preds.shape[0]
        src = scene2dataset[subset]
        print(f'evaluating camera_relocalization task on {src} {subset} subset, size at {subset_size}')

        pos_error_scene, orient_error_scene = camera_relocalization_errors(all_preds, all_gts)
        if src == 'CambridgeLandmarks':
            position_error_cl.append(pos_error_scene.mean())
            orientation_error_cl.append(orient_error_scene.mean())
        elif src == '7Scenes':
            position_error_7s.append(pos_error_scene.mean())
            orientation_error_7s.append(orient_error_scene.mean())
        else:
            raise NotImplementedError
    
    # average on scenes
    position_error_cl = np.mean(position_error_cl)
    orientation_error_cl = np.mean(orientation_error_cl)
    position_error_7s = np.mean(position_error_7s)
    orientation_error_7s = np.mean(orientation_error_7s)
    print(f'Trans on CL: mean = {position_error_cl:.3f}')
    print(f'Orient on CL: mean = {orientation_error_cl:.3f}')
    print(f'Trans on 7S: mean = {position_error_7s:.3f}')
    print(f'Orient on 7S: mean = {orientation_error_7s:.3f}')
    
    # average on data sources and transform to score
    task_score = (
        np.exp(-1.386*position_error_cl) + np.exp(-1.386*orientation_error_cl)
        + np.exp(-1.386*position_error_7s) + np.exp(-1.386*orientation_error_7s)
    ) / 4 * 100
    return task_score


def EvalRec(pred_h5py, gt_h5py):
    all_preds = np.array(pred_h5py['3d_reconstruction']['test'])
    all_gts = np.array(gt_h5py['3d_reconstruction']['test'])
    subset_size = all_preds.shape[0]
    print(f'evaluating 3d_reconstruction task on ShapeNet test subset, size at {subset_size}')

    area_intersect = np.logical_and(all_preds, all_gts).reshape(subset_size, -1)
    area_union = np.logical_or(all_preds, all_gts).reshape(subset_size, -1)
    
    iou = area_intersect.sum(-1) / area_union.sum(-1).mean()
    task_score = iou * 100
    print(f'Rec IoU: {task_score:.2f}')
    return task_score


def EvalRetrieval(pred_h5py, gt_h5py):
    all_preds = np.array(pred_h5py['vl_retrieval']['test'])
    all_gts = np.array(gt_h5py['vl_retrieval']['test'])
    subset_size = all_preds.shape[0]
    print(f'evaluating vl_retrieval task on Flickr30k test subset, size at {subset_size}')
    highest_ranks = []
    for i in range(subset_size):
        txt_gt_ids = all_gts[i]
        # get highest rank among 5 candidates
        for idx, txt_id in enumerate(all_preds[i]):
            if txt_id in txt_gt_ids:
                break
        highest_ranks.append(idx)
    
    highest_ranks = np.array(highest_ranks)
    recall_1 = (highest_ranks < 1).astype(float).mean() * 100
    recall_5 = (highest_ranks < 5).astype(float).mean() * 100
    recall_10 = (highest_ranks < 10).astype(float).mean() * 100
    print(f'Recall@1: {recall_1:.2f}, Recall@5: {recall_5:.2f}, Recall@10: {recall_10:.2f}')
    task_score = (recall_1+recall_5+recall_10) / 3
    return task_score


def EvalBbox(pred_h5py, gt_h5py):
    subsets = ['val', 'testA', 'testB']
    acc_all = []
    for subset in subsets:
        all_preds = np.array(pred_h5py['phrase_grounding'][subset])
        all_gts = np.array(gt_h5py['phrase_grounding'][subset])
        subset_size = all_preds.shape[0]
        print(f'evaluating phrase_grounding task on RefCOCO {subset} subset, size at {subset_size}')
        
        iou = box_iou(all_preds, all_gts)
        acc_all.append((iou >= 0.5).astype(float).mean())
    
    task_score = np.mean(acc_all) * 100
    print(f'Bbox Acc@0.5: {task_score:.2f}')
    return task_score


def EvalSeg(pred_h5py, gt_h5py):
    all_preds = np.array(pred_h5py['segmentation']['val'])
    all_gts = np.array(gt_h5py['segmentation']['val'])
    subset_size = all_preds.shape[0]
    print(f'evaluating segmentation task on ADE20k val subset, size at {subset_size}')

    num_classes = 151
    area_intersect, area_union = intersect_and_union(all_preds, all_gts, num_classes, ignore_index=0)[:2]
    
    iou = area_intersect / area_union
    task_score = np.nanmean(iou) * 100
    print(f'mIoU: {task_score:.2f}')
    return task_score


def EvalQA(pred_h5py, gt_h5py):
    all_preds = np.array(pred_h5py['vqa']['testdev'])
    all_gts = np.array(gt_h5py['vqa']['testdev'])
    subset_size = all_preds.shape[0]
    print(f'evaluating vqa task on GQA test-dev subset, size at {subset_size}')
    
    acc = (all_preds==all_gts).astype(float).mean()
    task_score = acc * 100
    print(f'Answer Acc: {task_score:.2f}')
    return task_score


def EvalVCR(pred_h5py, gt_h5py):
    all_preds = np.array(pred_h5py['common_sense']['val'])
    all_gts = np.array(gt_h5py['common_sense']['val'])
    subset_size = all_preds.shape[0]
    print(f'evaluating common_sense task on VCR val subset, size at {subset_size}')
    
    acc = (all_preds==all_gts).astype(float).mean()
    task_score = acc * 100
    print(f'Answer Acc: {task_score:.2f}')
    return task_score


def EvalBongard(pred_h5py, gt_h5py):
    all_preds = np.array(pred_h5py['bongard']['test'])
    all_gts = np.array(gt_h5py['bongard']['test'])
    subset_size = all_preds.shape[0]
    print(f'evaluating bongard task on Bongard-HOI test subset, size at {subset_size}')

    acc = (all_preds==all_gts).astype(float).mean()
    task_score = acc * 100
    print(f'Answer Acc: {task_score:.2f}')
    return task_score


def EvalNav(pred_h5py, gt_h5py=None):
    subsets = {'val_seen': 1021, 'val_unseen': 2349}
    scores = []
    for subset, subset_size in subsets.items():
        print(f'evaluating navigation task on R2R {subset} subset, size at {subset_size}')
        scores.append(np.array(pred_h5py['navigation'][f'{subset}-spl']))

    task_score = np.mean(scores) * 100
    print(f'SPL: {task_score:.2f}')
    return task_score
    

def EvalManip(pred_h5py, gt_h5py=None):
    subsets = [
        'assembling-kits-seq-unseen-colors', 'packing-unseen-google-objects-group',
        'put-block-in-bowl-unseen-colors', 'stack-block-pyramid-seq-unseen-colors',
        'packing-unseen-google-objects-seq', 'packing-boxes-pairs-unseen-colors',
        'separating-piles-unseen-colors', 'towers-of-hanoi-seq-unseen-colors'
    ]
    scores = []
    for subset in subsets:
        rewards = np.array(pred_h5py['manipulation'][f'{subset}-rewards'])
        print(f'evaluating manipulation task on Ravens {subset} subset, size at {rewards.shape[0]}')
        scores.append(np.mean(rewards))
    
    task_score = np.mean(scores) * 100
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
    position = output[:, :3]
    orientation = output[:, -4:]
    position_target = target[:, :3]
    orientation_target = target[:, -4:]

    position_error = np.linalg.norm(position-position_target, axis=-1)

    orientation = orientation / np.linalg.norm(orientation, axis=-1, keepdims=True)
    orientation_target = orientation_target / np.linalg.norm(orientation_target, axis=-1, keepdims=True)
    orientation_error = [quaternion_angular_error(p, t) for p, t in zip(orientation, orientation_target)]
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
