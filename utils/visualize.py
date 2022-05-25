import os
import torch
import numpy as np
import skimage.io as skio
import cv2
from . import io
from .html_writer import HtmlWriter
from torchvision.utils import draw_segmentation_masks
from fvcore.common.registry import Registry
VISUALIZE = Registry('Visualize')
norm_means = torch.as_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
norm_stds = torch.as_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@VISUALIZE.register()
@torch.no_grad()
def VisVLMatching(html_writer, model, dataloader, cfg, step, vis_dir):
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
        B = len(targets)

        add_special_token = (cfg.task.key == 'common_sense')
        outputs = model(imgs, txts, expand_batch=True, add_special_token=add_special_token)
        # [4B, 1] -> [B, 4]
        outputs = outputs.view(B, -1)
        preds = torch.topk(outputs, k=1, dim=-1).indices.squeeze()   # [B]

        for i in range(B):
            if count+i >= cfg.training.num_vis_samples:
                finish_vis = True
                break
            
            vis_img = imgs[i].mul_(norm_stds).add_(norm_means)
            vis_img = vis_img.detach().cpu().numpy() * 255
            vis_img = vis_img.astype(np.uint8).transpose(1, 2, 0)

            gt = targets[i]
            pred = preds[i]

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


@VISUALIZE.register()
@torch.no_grad()
def VisSeg(html_writer, model, dataloader, cfg, step, vis_dir):
    html_writer.add_element({
        0: 'query',
        1: 'visualization',
        2: 'prediction',
        3: 'ground truth'
    })
    count = 0
    finish_vis = False
    model.eval()
    num_classes = cfg.task.num_classes
    for data in dataloader:
        imgs, txts, targets = data
        outputs = model(imgs, txts=None)
        # [B, num_classes, H, W]
        refer = torch.arange(num_classes)[None, :, None, None]
        preds = (outputs.argmax(1, keepdim=True).repeat(1, num_classes, 1, 1).detach().cpu() == refer)
        gts = (targets.unsqueeze(1).repeat(1, num_classes, 1, 1).detach().cpu() == refer)

        B = len(targets)
        for i in range(B):
            if count+i >= cfg.training.num_vis_samples:
                finish_vis = True
                break
            
            vis_img = imgs[i].mul_(norm_stds).add_(norm_means)
            vis_img = (vis_img.detach().cpu() * 255).to(torch.uint8)
            vis_img_numpy = vis_img.numpy().transpose(1, 2, 0)
            vis_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '.png'
            skio.imsave(os.path.join(vis_dir, vis_name), vis_img_numpy)

            seg_pred = draw_segmentation_masks(image=vis_img, masks=preds[i])
            seg_pred = seg_pred.numpy().transpose(1, 2, 0)
            pred_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '_pred.png'
            skio.imsave(os.path.join(vis_dir, pred_name), seg_pred)

            seg_gt = draw_segmentation_masks(image=vis_img, masks=gts[i])
            seg_gt = seg_gt.numpy().transpose(1, 2, 0)
            gt_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '_gt.png'
            skio.imsave(os.path.join(vis_dir, gt_name), seg_gt)

            html_writer.add_element({
                0: txts[i],
                1: html_writer.image_tag(vis_name),
                2: html_writer.image_tag(pred_name),
                3: html_writer.image_tag(gt_name)
            })
        
        if finish_vis is True:
            break
        
        count += B


@VISUALIZE.register()
@torch.no_grad()
def VisDepth(html_writer, model, dataloader, cfg, step, vis_dir):
    html_writer.add_element({
        0: 'query',
        1: 'visualization',
        2: 'prediction',
        3: 'ground truth'
    })
    min_depth, max_depth = 1e-3, 10
    count = 0
    finish_vis = False
    model.eval()
    for data in dataloader:
        imgs, txts, targets = data
        outputs = model(imgs, txts=None)
        preds = (outputs.sigmoid().squeeze() * max_depth).clip(min_depth, max_depth)
        targets = targets.squeeze().clip(min_depth, max_depth)

        B = len(targets)
        for i in range(B):
            if count+i >= cfg.training.num_vis_samples:
                finish_vis = True
                break
            
            vis_img = imgs[i].mul_(norm_stds).add_(norm_means)
            vis_img = vis_img.detach().cpu().numpy() * 255
            vis_img = vis_img.astype(np.uint8).transpose(1, 2, 0)
            vis_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '.png'
            skio.imsave(os.path.join(vis_dir, vis_name), vis_img)

            depth_pred = preds[i] / max_depth
            depth_pred = depth_pred.detach().cpu().numpy() * 255
            depth_pred = depth_pred.astype(np.uint8)
            pred_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '_pred.png'
            skio.imsave(os.path.join(vis_dir, pred_name), depth_pred)

            depth_gt = targets[i] / max_depth
            depth_gt = depth_gt.detach().cpu().numpy() * 255
            depth_gt = depth_gt.astype(np.uint8)
            gt_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '_gt.png'
            skio.imsave(os.path.join(vis_dir, gt_name), depth_gt)

            html_writer.add_element({
                0: txts[i],
                1: html_writer.image_tag(vis_name),
                2: html_writer.image_tag(pred_name),
                3: html_writer.image_tag(gt_name)
            })
        
        if finish_vis is True:
            break
        
        count += B


@VISUALIZE.register()
@torch.no_grad()
def VisQA(html_writer, model, dataloader, cfg, step, vis_dir):
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
        outputs = model(imgs, txts)
        preds = torch.topk(outputs, k=1, dim=-1).indices.squeeze()

        B = len(targets)
        for i in range(B):
            if count+i >= cfg.training.num_vis_samples:
                finish_vis = True
                break
            
            vis_img = imgs[i].mul_(norm_stds).add_(norm_means)
            vis_img = vis_img.detach().cpu().numpy() * 255
            vis_img = vis_img.astype(np.uint8).transpose(1, 2, 0)

            gt = dataloader.dataset.idx_to_answer[targets[i]]
            pred = dataloader.dataset.idx_to_answer[
                preds[i].item() if isinstance(preds[i], torch.Tensor) else preds[i]
            ]

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


def visualize(model, dataloader, cfg, step, subset):
    vis_dir = os.path.join(
        cfg.exp_dir,
        f'visualizations/{subset}_'+str(step).zfill(6))
    io.mkdir_if_not_exists(vis_dir, recursive=True)

    html_writer = HtmlWriter(os.path.join(vis_dir, 'index.html'))
    VISUALIZE.get(cfg.task.visualize)(html_writer, model, dataloader, cfg, step, vis_dir)
    html_writer.close()
