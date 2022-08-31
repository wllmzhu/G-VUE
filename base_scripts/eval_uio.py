""" evaluation of Unified-IO and OFA, as additional baselines of large general-purpose models """

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.base import create_dataset
from models.uio import runner
from utils.misc import collate_fn
from tqdm import tqdm
from einops import rearrange
from absl import logging
logging.set_verbosity(logging.INFO)

norm_means = torch.as_tensor([0.485, 0.456, 0.406])
norm_stds = torch.as_tensor([0.229, 0.224, 0.225])
rgb2grey_coef = np.array([0.299, 0.587, 0.114])


def undo_img_transform(img):
    # transform a single image tensor [3, H, W] returned from __getitem__ to original array [H, W, 3]
    # as well as revert value normalization
    img = rearrange(img, 'C H W -> H W C')
    img = img.mul_(norm_stds).add_(norm_means)
    return img.cpu().numpy()


def compute_depth_errors(pred, gt):
    """ here pred and gt are both numpy.ndarray """
    delta = np.maximum((pred / gt), (gt / pred))
    d1 = (delta < 1.25).astype(np.float).mean()

    abs_rel = (np.abs(pred - gt) / gt).mean()

    rms = (pred - gt) ** 2
    rms = np.sqrt(rms.mean())

    return np.array([d1, abs_rel, rms])


def eval_depth(cfg, model, uio=True):
    if 'scratch' in cfg.task.dataset.info.train_img_dir:
        cfg.task.dataset.info.test_img_dir = '/mnt/huangjiangyong/NYUv2/labelled'
        cfg.task.dataset.info.test_index = '/mnt/huangjiangyong/NYUv2/sync/index_test.txt'
    dataset = create_dataset(cfg, 'test')
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    min_depth, max_depth = 1e-3, 10
    metrics = []
    logging.info(f'evaluating {"Unified-IO" if uio else "OFA"} on depth')
    for batch in tqdm(dataloader):
        # batch: (imgs, txts, targets)
        img, target = batch[0][0], batch[-1][0]
        if uio:
            img = undo_img_transform(img)
            out = model.depth(img)
        else:
            pass

        pred = (rgb2grey_coef * out['rescaled_image']).sum(-1)
        pred = (pred * max_depth).clip(min_depth, max_depth)
        
        target = target.cpu().squeeze().numpy()
        valid_mask = np.logical_and(target>min_depth, target<max_depth)
        metrics.append(compute_depth_errors(pred[valid_mask], target[valid_mask]).reshape(1, -1))
    
    metrics = np.concatenate(metrics, axis=0).mean(0)
    for i, k in enumerate(['d1', 'abs_rel', 'rms']):
        logging.info(f'{k}: {metrics[i]:.4f}')

    err2per = np.exp(-1.386*metrics[1:])
    task_score = (metrics[0]+err2per.sum()) / 3 * 100
    logging.info(f'task score: {task_score:.2f}\n')
    return task_score


OBJ = 'uio'
@hydra.main(config_path='../configs', config_name='base')
def main(cfg):
    uio = False
    if OBJ == 'uio':
        uio = True
        model = runner.ModelRunner('xl', '/mnt/huangjiangyong/UIO/xl_1000k.bin')
        # model.model.eval()   # ModelRunner.model is nn.Module rather than runner itself, flax.linen.Module has no eval()
    elif OBJ == 'ofa':
        pass
    else:
        raise ValueError('model not exist or prepared')
    
    if cfg.task.key == 'depth':
        eval_depth(cfg, model=model, uio=uio)


"""
take depth for example, run:
    python base_scripts/eval_uio.py task=depth output_dir=/mnt/huangjiangyong/GVUE
"""
if __name__ == '__main__':
    main()
