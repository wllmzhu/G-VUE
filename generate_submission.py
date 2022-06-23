import os
import hydra
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.base import JointModel
from datasets.base import create_dataset
from utils.misc import collate_fn
import h5py
from inference_fullset import inference

all_tasks = [
    'depth', 'camera_relocalization', '3d_reconstruction', 'vl_retrieval', 'phrase_grounding',
    'segmentation', 'vqa', 'common_sense', 'bongard', 'navigation', 'manipulation'
]


def init_model(cfg, device, task):
    model = JointModel(cfg.model).to(device)
    state_dict = model.state_dict()
    ckpt = torch.load(ckpt_path[task], map_location=device)
    loaded_params = 0
    for k, v in ckpt['model'].items():
        if k in state_dict and state_dict[k].size() == v.size():
            state_dict[k] = v
            loaded_params += 1

    model.load_state_dict(state_dict)
    print(f'loaded {loaded_params} parameters from {ckpt_path[task]} for {task} task')
    return model


def generate_h5py(cfg):
    device = f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu'

    h5py_file = h5py.File('preds_to_submit.h5py', 'w')
    
    for task in all_tasks:
        model = init_model(cfg, device, task)
        print(f'generating predictions on {task} task, {task_dataset} dataset')

        for subset in task_subsets:
            if subset != 'train':
                dataset = create_dataset(cfg, subset)
                l = len(datasets[subset])
                print(f'{subset} set size: {l}')

                dataloader = dataloader = DataLoader(
                    dataset,
                    batch_size=cfg.eval.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=cfg.eval.num_workers,
                    pin_memory=True
                )
                h5py_file = inference(task)(model, dataloader, h5py_file)
    
    # TODO: Act tasks

    h5py_file.close()


@hydra.main(config_path='./configs', config_name='base')
def main(cfg):
    if cfg.task.key == 'vl_retrieval':
        cfg.eval.batch_size = 1
        cfg.eval.num_workers = 0
    elif cfg.task.key == 'bongard':
        cfg.eval.batch_size = 32
        cfg.eval.num_workers = 8
    elif cfg.task.key == '3d_reconstruction':
        cfg.eval.batch_size = 50
    
    generate_h5py(cfg)
    

if __name__=='__main__':
    main()
