import os
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.metrics import build_evaluator
from models.base import JointModel
from datasets.base import create_dataset
from utils.misc import collate_fn
import utils.io as io


def eval_non_train_full(cfg):
    device = f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu'

    datasets = {}
    max_size = 0
    for subset in cfg.task.dataset.info.subsets:
        if subset != 'train':
            datasets.update({
                subset: create_dataset(cfg, subset)
            })
            l = len(datasets[subset])
            print(f'{subset} set size: {l}')
            max_size = max(l, max_size)
    cfg.eval.num_val_samples = max_size

    print(OmegaConf.to_yaml(cfg))

    model = JointModel(cfg.model).to(device)
    state_dict = model.state_dict()
    ckpt = torch.load(cfg.eval.ckpt, map_location=device)
    for k, v in ckpt['model'].items():
        if k in state_dict and state_dict[k].size() == v.size():
            state_dict[k] = v
            print(f'loaded {k}')

    model.load_state_dict(state_dict)
    print(f'Loading checkpoint at the end of epoch {ckpt["epoch"]}')
    
    evaluator = build_evaluator(cfg.task.metrics)
    print(f'Exp: {cfg.exp_name}')
    for subset, dataset in datasets.items():
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.eval.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.eval.num_workers,
            pin_memory=True
        )
        dataset_name = cfg.task.dataset.key
        print(f'Evaluating on {dataset_name}')

        metrics = evaluator(model, dataloader, cfg)
        
        eval_str = f'Dataset: {dataset_name} | Subset: {subset}'

        if len(metrics.keys()) > 0:
            for k, v in metrics.items():
                v = round(v, 4)
                eval_str += f' | {k}: {v}'

        print(eval_str)


@hydra.main(config_path='./configs', config_name='base')
def main(cfg):
    eval_non_train_full(cfg)
    

if __name__=='__main__':
    main()
