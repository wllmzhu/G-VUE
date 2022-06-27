import os
import hydra
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from datasets.base import create_dataset
from utils.misc import collate_fn
from task_evaluation_lib import evaluate
from collections import OrderedDict
from models.nav_decoder.eval import R2REvaluation

subsets = {
    'depth': ['test'], 'camera_relocalization': ['test'], '3d_reconstruction': ['test'],
    'vl_retrieval': ['test'], 'phrase_grounding': ['val', 'testA', 'testB'], 'segmentation': ['val'],
    'vqa': ['testdev'], 'common_sense': ['val'], 'bongard': ['test'],
    'navigation': ['val_seen', 'val_unseen'],
    'manipulation': [
        'assembling-kits-seq-unseen-colors', 'packing-unseen-google-objects-group',
        'put-block-in-bowl-unseen-colors', 'stack-block-pyramid-seq-unseen-colors',
        'packing-unseen-google-objects-seq', 'packing-boxes-pairs-unseen-colors',
        'separating-piles-unseen-colors', 'towers-of-hanoi-seq-unseen-colors'
    ]
}


def evaluate_camera_pose(cfg, h5py_file):
    task_scores = []
    data_types = ['cambridgelandmarks', 'sevenscenes']
    for data_type in data_types:
        cfg.task.dataset.info.dataset = data_type
        for scene in cfg.task.dataset.info[data_type].all_scenes:
            cfg.task.dataset.info[data_type].scene = scene
            print(f'evaluating results on camera_relocalization task, {data_type} dataset, {scene} scene')
            for subset in subsets['camera_relocalization']:
                dataset = create_dataset(cfg, subset)
                print(f'{subset} set size: {len(dataset)}')
                dataloader = DataLoader(
                    dataset,
                    batch_size=cfg.eval.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=cfg.eval.num_workers,
                    pin_memory=True
                )
                task_scores.append(evaluate('camera_relocalization')(dataloader, h5py_file))
    
    return task_scores


def evaluate_nav(h5py_file):
    task_scores = []
    featurized_scans = h5py_file['navigation/featurized_scans']
    val_envs = OrderedDict(
        (split, R2REvaluation(cfg, [split], featurized_scans, tok)) for split in subsets['navigation'])
    for env_name, evaluator in val_envs.items():
        task_scores.extend(evaluate('navigation')(env_name, evaluator, h5py_file))
    return task_scores

def evaluate_manip(h5py_file):
    task_scores = []
    for eval_task in subsets['manipulation']:
        task_scores.append(evaluate('manipulation')(eval_task, h5py_file))
    return task_scores


def evaluate_h5py(cfg, h5py_file):
    if cfg.task.key == 'camera_relocalization':
        task_scores = evaluate_camera_pose(cfg, h5py_file)
    elif cfg.task.key == 'navigation':
        task_scores = evaluate_nav(h5py_file)
    elif cfg.task.key == 'manipulation':
        task_scores = evaluate_manip(h5py_file)
    else:
        print(f'evaluating results on {cfg.task.key} task, {cfg.task.dataset.key} dataset')
        task_scores = []
        for subset in subsets[cfg.task.key]:
            dataset = create_dataset(cfg, subset)
            print(f'{subset} set size: {len(dataset)}')
            dataloader = DataLoader(
                dataset,
                batch_size=cfg.eval.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=cfg.eval.num_workers,
                pin_memory=True
            )
            task_scores.append(evaluate(cfg.task.key)(dataloader, h5py_file))
    
    score = f'{np.mean(task_scores):.2f}'
    print(f'{cfg.task.key} task score: {score}')
        


@hydra.main(config_path='./configs', config_name='base')
def main(cfg):
    h5py_file = h5py.File('submission.h5py', 'r')
    evaluate_h5py(cfg, h5py_file)
    h5py_file.close()
    

if __name__=='__main__':
    main()
