import os
import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import skimage.io as skio
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from pytorch_transformers.optimization import WarmupLinearSchedule

from models.base import JointModel
from datasets.base import create_dataset
from utils.html_writer import HtmlWriter
from utils.misc import collate_fn
import utils.io as io


def grad_norm(params):
    total_norm = 0
    for p in params:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    
    return total_norm ** (1. / 2)
    

def visualize(model, dataloader, cfg, step, subset):
    vis_dir = os.path.join(
        cfg.exp_dir,
        f'visualizations/{subset}_'+str(step).zfill(6))
    io.mkdir_if_not_exists(vis_dir, recursive=True)

    html_writer = HtmlWriter(os.path.join(vis_dir, 'index.html'))
    html_writer.add_element({
        0: 'query',
        1: 'visualization',
        2: 'prediction',
        3: 'ground truth',
        4: 'probabilities'
    })
    count = 0
    finish_vis = False
    model.eval()
    for data in dataloader:
        pass
    
    html_writer.close()


def get_lrs(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    
    return lrs


def train_worker(gpu, cfg):
    cfg.gpu = gpu
    if cfg.gpu is not None:
        print(f'Use GPU: {cfg.gpu} for training')
    device = f'cuda:{cfg.gpu}'

    if gpu == 0:
        print(OmegaConf.to_yaml(cfg))

    datasets = {
        'train': create_dataset(cfg, 'train'),
        'val': create_dataset(cfg, 'val')
    }
    for subset, dataset in datasets.items():
        print(f'{subset} set size:', len(dataset))

    model = JointModel(cfg.model)

    if cfg.multiprocessing_distributed:
        cfg.rank = cfg.rank * cfg.ngpus_per_node + cfg.gpu

        torch.cuda.set_device(cfg.gpu)
        
        dist.init_process_group(
            backend=cfg.dist_backend, 
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank)

        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.gpu], find_unused_parameters=True)

        # create sampler for dataloader
        sampler = {'val': None}
        sampler['train'] = torch.utils.data.distributed.DistributedSampler(
            datasets['train'], shuffle=True)
    else:
        model.to(device)
        sampler = {'train': None, 'val': None}

    dataloaders = {}
    for subset, dataset in datasets.items():
        dataloaders[subset] = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            shuffle=(sampler[subset] is None),
            sampler=sampler[subset])

    if gpu == 0:
        writer = SummaryWriter(log_dir=cfg.tb_dir)

    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)

    print(f'collect {len(params)} parameters for optimization')

    optimizer = torch.optim.AdamW(params, lr=cfg.training.lr, betas=cfg.training.betas)

    step = 0
    last_epoch = -1
    model_selection_metric = 0
    best_metric = 0
    best_epoch = -1
    if os.path.exists(cfg.training.ckpt):
        ckpt = torch.load(cfg.training.ckpt, map_location=device)
        state_dict = model.state_dict()
        for k, v in ckpt['model'].items():
            if k in state_dict and state_dict[k].size() == v.size():
                v.requires_grad = state_dict[k].requires_grad
                state_dict[k] = v
                print(f'loaded {k}')

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer'])

        step = ckpt['step']
        last_epoch = ckpt['epoch']
        if model_selection_metric in ckpt:
            model_selection_metric = ckpt['model_selection_metric']
        else:
            model_selection_metric = 0
            
        best_metric = model_selection_metric
        best_epoch = last_epoch
        print(f'Loading checkpoint at the end of epoch {last_epoch}')
    
    warmup_iters = len(dataloaders['train'])
    if cfg.training.lr_warmup is True:
        if cfg.training.lr_linear_decay:
            num_train_optimization_steps = len(dataloaders['train']) * cfg.training.num_epochs
            warmup_steps = cfg.training.lr_warmup_fraction * num_train_optimization_steps
            warmup_scheduler = WarmupLinearSchedule(
                optimizer,
                warmup_steps=warmup_steps,
                t_total=num_train_optimization_steps,
                last_epoch=-1)
        else:
            warmup_scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=warmup_iters,
                last_epoch=-1)   # updated every iter not epoch
            if gpu == 0:
                print('Warmup iters:', warmup_iters)

        if os.path.exists(cfg.training.ckpt):
            warmup_scheduler.load_state_dict(ckpt['warmup_scheduler'])

    if cfg.training.lr_warmup and not cfg.training.lr_linear_decay:
        # zero grad step needed for warmup scheduler
        optimizer.zero_grad()
        optimizer.step()

    training_epochs = cfg.training.num_epochs

    for epoch in range(last_epoch+1, training_epochs):

        if cfg.multiprocessing_distributed:
            sampler['train'].set_epoch(epoch)

        for it, data in enumerate(dataloaders['train']):
            model.train()
            loss = model()

            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                if cfg.training.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        params, cfg.training.clip_max_norm
                    )
                optimizer.step()
            
            if gpu == 0 and step % cfg.training.log_step == 0:
                loss_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                if cfg.training.lr_linear_decay:
                    loss_str += f' LR: {warmup_scheduler.get_last_lr()[0]} | '
                loss_value = round(loss.item(), 4)
                writer.add_scalar('Epoch', epoch, step)
                writer.add_scalar('Iter', it, step)
                writer.add_scalar('Step', step, step)
                writer.add_scalar('Best Epoch', best_epoch, step)
                for j, group_lr in enumerate(get_lrs(optimizer)):
                    writer.add_scalar(
                        f'Lr/optimizer/group_{j}',
                        group_lr,
                        step
                    )
                
                loss_value = round(loss_value, 4)
                loss_str += f'Loss: {loss_value} | '
                writer.add_scalar(f'Loss/train', loss_value, step)
                print(loss_str)

            if gpu == 0 and step % cfg.training.vis_step == 0:
                with torch.no_grad():
                    for subset in ['train', 'val']:
                        print(f'Visualizing {subset} ...')
                        visualize(model, dataloaders[subset], cfg, step, subset)

            if gpu == 0 and step % (10*cfg.training.log_step) == 0:
                print('Exp:', cfg.exp_name)
                
            step += 1

            if cfg.training.lr_linear_decay:
                warmup_scheduler.step()
            elif cfg.training.lr_warmup is True and epoch == 0 and it < warmup_iters:
                warmup_scheduler.step(it)
        
        if gpu == 0:
            model_selection_metric = 0
            # for eval_subset in ['train', 'val']:
            for eval_subset in ['val']:
                for dataset_name in dataloaders[eval_subset].dataset.datasets:
                    if 'refcoco' not in dataset_name:
                        continue
                    print(f'Evaluating on {dataset_name}')
                    eval_dataset = dataloaders[eval_subset].dataset.datasets[dataset_name]
                    eval_dataloader = DataLoader(
                        eval_dataset,
                        batch_size=cfg.eval.batch_size,
                        num_workers=cfg.eval.num_workers,
                        shuffle=True,
                        collate_fn=collate_fn)
                    
                    with torch.no_grad():
                        metrics = None
                    
                    eval_str = f'Dataset: {dataset_name} | Subset: {eval_subset} | Epoch: {epoch} | '

                    react_rate = round(metrics['reaction_rate'], 4)
                    eval_str += f'reaction rate: {react_rate} | '
                    writer.add_scalar(f'{eval_subset}/{dataset_name}/reaction_rate', react_rate, step)

                    bbox_AP50 = 0 
                    bbox_mAP = 0 
                    mask_mIoU = 0
                    mask_AP = [0, 0, 0]
                    depth_l1_error = 0
                    if metrics['bbox_AP@0.5'] is not None:
                        bbox_AP50 = round(metrics['bbox_AP@0.5'], 4)
                        bbox_mAP = round(metrics['bbox_mAP'], 4)
                        eval_str += f'bbox AP@0.5: {bbox_AP50} | bbox mAP: {bbox_mAP}'
                        writer.add_scalar(f'{eval_subset}/{dataset_name}/AP@0.5', bbox_AP50, step)
                        writer.add_scalar(f'{eval_subset}/{dataset_name}/mAP', bbox_mAP, step)
                    if metrics['mask_mIoU'] is not None:
                        mask_mIoU = round(metrics['mask_mIoU'], 4)
                        mask_AP = metrics['mask_AP']
                        mask_AP = [round(x, 4) for x in mask_AP]
                        eval_str += f'mask mIoU: {mask_mIoU} | mask AP: {mask_AP}'
                        writer.add_scalar(f'{eval_subset}/{dataset_name}/mIoU', mask_mIoU, step)
                        writer.add_scalar(f'{eval_subset}/{dataset_name}/AP@0.5', mask_AP[0], step)
                        writer.add_scalar(f'{eval_subset}/{dataset_name}/AP@0.7', mask_AP[1], step)
                        writer.add_scalar(f'{eval_subset}/{dataset_name}/AP@0.9', mask_AP[2], step)
                    if metrics['depth_l1_error'] is not None:
                        depth_l1_error = round(metrics['depth_l1_error'], 4)
                        eval_str += f'depth l1 error: {depth_l1_error}'
                        writer.add_scalar(f'{eval_subset}/{dataset_name}/l1_error', depth_l1_error, step)
                    
                    print(eval_str)

                    if eval_subset == 'val':
                        model_selection_metric = model_selection_metric + \
                                                 bbox_AP50 + bbox_mAP + mask_mIoU + \
                                                 mask_AP[0] + mask_AP[1] + mask_AP[2] + \
                                                 - depth_l1_error

            if model_selection_metric > best_metric:
                print('Saving checkpoint ...')
                best_metric = model_selection_metric
                best_epoch = epoch
                writer.add_scalar('Best Epoch', best_epoch, step)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': best_epoch,
                    'step': step,
                    'model_selection_metric': model_selection_metric,
                    'warmup_scheduler': warmup_scheduler.state_dict() if cfg.training.lr_linear_decay else None,
                }, os.path.join(cfg.ckpt_dir, 'model.pth'))


@hydra.main(config_path='./config', config_name='vito')
def main(cfg):
    io.mkdir_if_not_exists(cfg.ckpt_dir, recursive=True)
    io.mkdir_if_not_exists(cfg.tb_dir, recursive=True)

    if cfg.multiprocessing_distributed:
        cfg.world_size = cfg.ngpus_per_node * cfg.num_nodes
        cfg.training.batch_size = int(cfg.training.batch_size / cfg.ngpus_per_node)
        cfg.training.num_workers = int(
            (cfg.training.num_workers + cfg.ngpus_per_node - 1) / cfg.ngpus_per_node
        )

        mp.spawn(train_worker, nprocs=cfg.ngpus_per_node, args=(cfg,))
    else:
        train_worker(cfg.gpu, cfg)
    

if __name__=='__main__':
    main()
