import os
import torch
import numpy as np
import skimage.io as skio
import cv2
from . import io
from .html_writer import HtmlWriter
from torch.nn.functional import interpolate
from fvcore.common.registry import Registry
VISUALIZE = Registry('Visualize')
norm_means = torch.as_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
norm_stds = torch.as_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@VISUALIZE.register()
@torch.no_grad()
def VisBbox(html_writer, model, dataloader, cfg, step, vis_dir):
    html_writer.add_element({
        0: 'query',
        1: 'visualization',
        2: 'prediction',
        3: 'ground truth'
    })
    count = 0
    finish_vis = False
    model.eval()
    for data in dataloader:
        imgs, txts, targets = data
        outputs = model(imgs, txts).sigmoid()
        preds = torch.stack([
            outputs[:, 0]-outputs[:, 2]/2,
            outputs[:, 1]-outputs[:, 3]/2,
            outputs[:, 0]+outputs[:, 2]/2,
            outputs[:, 1]+outputs[:, 3]/2
        ], dim=1)

        B = len(targets)
        for i in range(B):
            if count+i >= cfg.training.num_vis_samples:
                finish_vis = True
                break
            
            vis_img = imgs[i].mul_(norm_stds).add_(norm_means)
            vis_img = vis_img.detach().cpu().numpy() * 255
            vis_img = vis_img.astype(np.uint8).transpose(1, 2, 0)

            gt = targets[i].detach().cpu().numpy()
            vis_img = vis_bbox(gt, vis_img, color=(0, 255, 0))
            pred = preds[i].detach().cpu().numpy()
            vis_img = vis_bbox(pred, vis_img, color=(0, 0, 255))

            vis_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '.png'
            skio.imsave(os.path.join(vis_dir, vis_name), vis_img)

            html_writer.add_element({
                0: txts[i],
                1: html_writer.image_tag(vis_name),
                2: pred,
                3: gt
            })
        
        if finish_vis is True:
            break
        
        count += B


def vis_bbox(bbox, img, color=(255, 0, 0)):
    # format: x1, y1, x2, y2
    im_h, im_w = img.shape[:2]

    x1, y1, x2, y2 = bbox * [im_w, im_h, im_w, im_h]
    x1 = max(0, min(x1, im_w-1))
    x2 = max(x1, min(x2, im_w-1))
    y1 = max(0, min(y1, im_h-1))
    y2 = max(y1, min(y2, im_h-1))
    
    img = np.ascontiguousarray(img)

    cv2.rectangle(img, [int(x1),int(y1)], [int(x2),int(y2)], color, 2)

    return img


def vis_mask(mask, img, color=(255, 0, 0), modify=False, alpha=0.2):
    if modify == True:
        img_ = img
    else:
        img_ = np.copy(img)
    
    # mask shape may not match img shape
    if mask.shape != img_.shape[:2]:
        # ndarray [256, 256] -> tensor [1, 1, 256, 256] -> ndarray [256, 256]
        mask = torch.from_numpy(mask)
        mask = interpolate(mask[None, None].float(), img_.shape[:2], mode="nearest")[0, 0]
        mask = mask.numpy()
    
    if mask.dtype != np.uint8:
        mask = np.clip(255*mask, 0, 255).astype(np.uint8)
    
    rr, cc = mask.nonzero()
    skdraw.set_color(img_, (rr, cc), color, alpha=alpha)   # area
    return img_, mask


def visualize(model, dataloader, cfg, step, subset):
    vis_dir = os.path.join(
        cfg.exp_dir,
        f'visualizations/{subset}_'+str(step).zfill(6))
    io.mkdir_if_not_exists(vis_dir, recursive=True)

    html_writer = HtmlWriter(os.path.join(vis_dir, 'index.html'))
    VISUALIZE.get(cfg.task.visualize)(html_writer, model, dataloader, cfg, step, vis_dir)
    html_writer.close()


def compute_iou_mask(pred_mask, gt_mask):
    """
    masks are both bool type
    """
    inter = np.sum(pred_mask*gt_mask)
    union = np.sum(pred_mask+gt_mask)
    iou = inter / (union+1e-6)
    return iou
