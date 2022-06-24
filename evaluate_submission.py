import os
import hydra
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from datasets.base import create_dataset
from utils.misc import collate_fn
from evaluate_all_tasks import evaluate

subsets = {
    'depth': ['test'], 'camera_relocalization': ['test'], '3d_reconstruction': ['test'],
    'vl_retrieval': ['test'], 'phrase_grounding': ['val', 'testA', 'testB'], 'segmentation': ['val'],
    'vqa': ['testdev'], 'common_sense': ['val'], 'bongard': ['test']
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
    
    print(f'{cfg.task.key} task score: {np.mean(task_scores):.2f}')
    return np.mean(task_scores)


def evaluate_h5py(cfg):
    h5py_file = h5py.File('preds_to_submit.h5py', 'r')

    if cfg.task.key == 'camera_relocalization':
        task_scores = evaluate_camera_pose(cfg, h5py_file)
    elif cfg.task.key == 'navigation':
        raise NotImplementedError
    elif cfg.task.key == 'manipulation':
        task_scores = evaluate_manip(cfg, )
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
    
    print(f'{cfg.task.key} task score: {np.mean(task_scores):.2f}')
    
    # TODO: Act tasks

    h5py_file.close()


@hydra.main(config_path='./configs', config_name='base')
def main(cfg):
    evaluate_h5py(cfg)
    

if __name__=='__main__':
    main()
