import os
import torch
import numpy as np
import skimage.io as skio
import cv2
from . import io
from math import ceil
from .html_writer import HtmlWriter
from torchvision.utils import draw_segmentation_masks
from einops import rearrange
from fvcore.common.registry import Registry
from models.rec3d_decoder.qual_util import sdf_to_mesh, save_mesh_as_gif, init_mesh_renderer
VISUALIZE = Registry('Visualize')
norm_means = torch.as_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
norm_stds = torch.as_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@VISUALIZE.register()
@torch.no_grad()
def VisBongard(html_writer, model, dataloader, cfg, step, vis_dir):
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

        outputs = model(imgs, txts, cfg.task.key)
        # [2B, 2] -> [2B]
        preds = torch.topk(outputs, k=1, dim=-1).indices.squeeze()

        targets = targets.flatten()

        imgs = rearrange(imgs, 'B r C H W -> (B r) C H W')

        for i in range(2*B):
            if count+i >= cfg.training.num_vis_samples:
                finish_vis = True
                break
            # imgs: [2B, 3x13, H, W]
            assert imgs.shape[1] == 39, "channels consist of 12 shot images and a query image"

            vis_imgs_list = []
            for j in range(0, 39, 3):
                # 12 shot imgs + 1 query image
                vis_img = imgs[i][j:j+3].mul_(norm_stds).add_(norm_means)
                vis_img = vis_img.detach().cpu().numpy() * 255
                vis_img = vis_img.astype(np.uint8).transpose(1, 2, 0)
                if j < 36:
                    vis_imgs_list.append(vis_img)
            
            # query image
            query_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '_query.png'
            skio.imsave(os.path.join(vis_dir, query_name), vis_img)
            
            # compose holistic visualization
            vis_imgs = np.concatenate([
                np.concatenate(vis_imgs_list[:3], axis=0),
                np.concatenate(vis_imgs_list[3:6], axis=0),
                np.zeros((3*vis_img.shape[0], 5, 3)),   # middle separating line
                np.concatenate(vis_imgs_list[6:9], axis=0),
                np.concatenate(vis_imgs_list[9:], axis=0)
            ], axis=1).astype(np.uint8)
            
            shot_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '_shot.png'
            skio.imsave(os.path.join(vis_dir, shot_name), vis_imgs)

            gt = targets[i]
            pred = preds[i]

            html_writer.add_element({
                0: html_writer.image_tag(query_name),
                1: html_writer.image_tag(shot_name),
                2: pred,
                3: gt
            })
        
        if finish_vis is True:
            break
        
        count += 2*B


@VISUALIZE.register()
@torch.no_grad()
def VisRetrieval(html_writer, model, dataloader, cfg, step, vis_dir):
    html_writer.add_element({
        0: 'query',
        1: 'visualization',
        2: 'prediction',
        3: 'ground truth'
    })
    count = 0
    finish_vis = False
    model.eval()

    if dataloader.dataset.subset == 'train':
        for data in dataloader:
            imgs, txts, targets = data
            B = len(targets)

            outputs = model(imgs, txts, cfg.task.key)
            _, preds = torch.sort(outputs, dim=-1, descending=True)
            preds = preds.detach().cpu()

            for i in range(B):
                if count+i >= cfg.training.num_vis_samples:
                    finish_vis = True
                    break

                vis_imgs = imgs[i].mul_(norm_stds).add_(norm_means)
                vis_imgs = vis_imgs.detach().cpu().numpy() * 255
                vis_imgs = vis_imgs.astype(np.uint8)
                
                pred = (preds[i]==i).nonzero()[:, 0].numpy()

                vis_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '.png'
                skio.imsave(os.path.join(vis_dir, vis_name), vis_imgs)

                html_writer.add_element({
                    0: txts[i],
                    1: html_writer.image_tag(vis_name),
                    2: pred,
                    3: 0
                })
            
            if finish_vis is True:
                break
            count += B
    
    else:
        txts = dataloader.dataset.texts
        img2txt_id = dataloader.dataset.img2txt
        step_size = cfg.task.text_batch_size
        for data in dataloader:
            imgs, _, targets = data
            B = len(targets)

            img2txt_scores = []
            num_step = ceil(len(txts)/step_size)
            for i in range(num_step):
                txt_batch = [txts[i*step_size:(i+1)*step_size]] * B
                outputs = model(imgs, txt_batch, cfg.task.key)
                img2txt_scores.append(outputs.view(B, -1))

            img2txt_scores = torch.cat(img2txt_scores, dim=-1)
            # [B, len(txts)]
            _, preds = torch.sort(img2txt_scores, dim=-1, descending=True)
            preds = preds.detach().cpu()

            for i in range(B):
                if count+i >= cfg.training.num_vis_samples:
                    finish_vis = True
                    break

                gt = targets[i]
                txt_gt_ids = img2txt_id[gt]   # 5 ground truth candidates
                pred_ranks = (preds[i].apply_(lambda x: x in txt_gt_ids)).nonzero()[:, 0]
                pred_ranks = pred_ranks.numpy()

                captions = [txts[j] for j in txt_gt_ids]

                vis_img = imgs[i].mul_(norm_stds).add_(norm_means)
                vis_img = vis_img.detach().cpu().numpy() * 255
                vis_img = vis_img.astype(np.uint8).transpose(1, 2, 0)

                vis_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '.png'
                skio.imsave(os.path.join(vis_dir, vis_name), vis_img)

                html_writer.add_element({
                    0: captions,
                    1: html_writer.image_tag(vis_name),
                    2: pred_ranks,
                    3: 0
                })
            
            if finish_vis is True:
                break
            count += B


@VISUALIZE.register()
@torch.no_grad()
def VisVCR(html_writer, model, dataloader, cfg, step, vis_dir):
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

        outputs = model(imgs, txts, cfg.task.key)
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


@VISUALIZE.register()
@torch.no_grad()
def VisRec(html_writer, model, dataloader, cfg, step, vis_dir):
    dist, elev, azim = 1.7, 20, 20
    mesh_renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device='cpu')

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
        tsdf = torch.stack([ele[1] for ele in targets]).cpu()
        B = tsdf.shape[0]

        img_logits = model(imgs.cuda())
        outputs = model.decoder.inference(img_logits).detach().cpu()

        for i in range(B):
            if count+i >= cfg.training.num_vis_samples:
                finish_vis = True
                break

            vis_img = imgs[i].mul_(norm_stds).add_(norm_means)
            vis_img = vis_img.detach().cpu().numpy() * 255
            vis_img = vis_img.astype(np.uint8).transpose(1, 2, 0)
            vis_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '.png'
            skio.imsave(os.path.join(vis_dir, vis_name), vis_img)

            pred_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '_pred.gif'
            gen_mesh = sdf_to_mesh(outputs[i:i+1])
            save_mesh_as_gif(mesh_renderer, gen_mesh, nrow=1, out_name=os.path.join(vis_dir, pred_name), device='cpu')

            gt_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '_gt.gif'
            gen_mesh = sdf_to_mesh(tsdf[i:i+1])
            save_mesh_as_gif(mesh_renderer, gen_mesh, nrow=1, out_name=os.path.join(vis_dir, gt_name), device='cpu')

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
def VisCameraRelocalization(html_writer, model, dataloader, cfg, step, vis_dir):
    """
    Just visualize input images
    """
    html_writer.add_element({
        0: 'query',
        1: 'visualization',
        2: 'prediction',
        3: 'ground truth'
    })
    count = 0
    finish_vis = False

    for data in dataloader:
        imgs, txts, targets = data
        B = imgs.shape[0]

        outputs = model(imgs)

        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        for i in range(B):
            if count+i >= cfg.training.num_vis_samples:
                finish_vis = True
                break

            vis_img = imgs[i].mul_(norm_stds).add_(norm_means)
            vis_img = vis_img.detach().cpu().numpy() * 255
            vis_img = vis_img.astype(np.uint8).transpose(1, 2, 0)
            vis_name = str(step).zfill(6) + '_' + str(count+i).zfill(4) + '.png'
            skio.imsave(os.path.join(vis_dir, vis_name), vis_img)

            html_writer.add_element({
                0: txts[i],
                1: html_writer.image_tag(vis_name),
                2: outputs[i],
                3: targets[i],
            })
        
        if finish_vis is True:
            break
        
        count += B

def visualize(model, dataloader, cfg, step, subset):
    vis_dir = os.path.join(
        cfg.exp_dir,
        f'visualizations/{subset}_'+str(step).zfill(6))
    io.mkdir_if_not_exists(vis_dir, recursive=True)

    html_writer = HtmlWriter(os.path.join(vis_dir, 'index.html'))
    VISUALIZE.get(cfg.task.visualize)(html_writer, model, dataloader, cfg, step, vis_dir)
    html_writer.close()
