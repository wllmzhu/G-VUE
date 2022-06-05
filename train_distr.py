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
from models.metrics import build_evaluator
from warmup_scheduler import GradualWarmupScheduler
from pytorch_transformers.optimization import WarmupLinearSchedule

from models.base import JointModel
from datasets.base import create_dataset
from utils.html_writer import HtmlWriter
from utils.misc import collate_fn
from utils.visualize import visualize
import utils.io as io


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

    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=cfg.training.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            shuffle=(sampler[subset] is None),
            sampler=sampler[subset]
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=cfg.eval.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.eval.num_workers,
            pin_memory=True,
            shuffle=(sampler[subset] is None),
            sampler=sampler[subset]
        )
    }

    if gpu == 0:
        writer = SummaryWriter(log_dir=cfg.tb_dir)

    if cfg.backbone.fix:
        params = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                params.append(p)
                print(f'add {n} {p.shape} for optimization')
        print(f'collect {len(params)} parameters for optimization')

        optimizer = torch.optim.AdamW(params, lr=cfg.training.lr, betas=cfg.training.betas)
    else:
        params1 = []
        params2 = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                if 'v_backbone' in n:
                    params2.append(p)
                    print(f'add {n} {p.shape} for type 2 optimization')
                else:
                    params1.append(p)
                    print(f'add {n} {p.shape} for type 1 optimization')
        print(f'collect {len(params1)} + {len(params2)} parameters for optimization')
        param_group = [
            {'params': params1, 'lr': cfg.training.lr},
            {'params': params2, 'lr': cfg.training.lr_backbone}
        ]
        optimizer = torch.optim.AdamW(param_group, betas=cfg.training.betas)

    step = 0
    last_epoch = -1
    best_metric = -100
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
        if 'model_selection_metric' in ckpt:
            best_metric = ckpt['model_selection_metric']
        
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

    evaluator = build_evaluator(cfg.task.metrics)

    for epoch in range(last_epoch+1, cfg.training.num_epochs):

        if cfg.multiprocessing_distributed:
            sampler['train'].set_epoch(epoch)

        for it, data in enumerate(dataloaders['train']):
            imgs, txts, targets = data
            if txts[0] is None:
                txts = None
            
            model.train()
            optimizer.zero_grad()

            outputs = model(imgs, txts, cfg.task.key)

            if cfg.task.key == '3d_reconstruction':
                targets = torch.stack([ele[0] for ele in targets])
            if not isinstance(targets, torch.Tensor):
                targets = torch.as_tensor(targets)
            loss = model.criterion(outputs, targets)
            loss.backward()

            if cfg.training.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    params if cfg.backbone.fix else params1+params2, cfg.training.clip_max_norm
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

            # if gpu == 0 and step % cfg.training.vis_step == 0:
            if gpu == 0 and step % cfg.training.vis_step == 0 and epoch == cfg.training.num_epochs-1:
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
            for eval_subset in ['val']:
                dataloader = dataloaders['val']
                dataset_name = cfg.task.dataset.key
                print(f'Evaluating on {dataset_name}')

                metrics = evaluator(model, dataloader, cfg)
                
                eval_str = f'Dataset: {dataset_name} | Subset: {eval_subset} | Epoch: {epoch}'

                if len(metrics.keys()) > 0:
                    for k, v in metrics.items():
                        if eval_subset != 'train':
                            if k not in cfg.eval.lower_better:
                                model_selection_metric += v
                            else:
                                model_selection_metric -= v
                        
                        v = round(v, 4)
                        eval_str += f' | {k}: {v}'
                        writer.add_scalar(f'{eval_subset}/{dataset_name}/{k}', v, step)

                print(eval_str)

            if model_selection_metric > best_metric:
                print('Saving checkpoint ...')
                best_metric = model_selection_metric
                best_epoch = epoch
                writer.add_scalar('Best Epoch', best_epoch, step)
                save_ckpt(
                    model=model, optimizer=optimizer.state_dict(), epoch=best_epoch, step=step,
                    metrics=model_selection_metric,
                    scheduler=warmup_scheduler.state_dict() if cfg.training.lr_linear_decay else None,
                    path=os.path.join(cfg.ckpt_dir, 'model.pth')
                )


def save_ckpt(model, optimizer, epoch, step, metrics, scheduler, path):
    sd = model.state_dict()
    for n, p in model.named_parameters():
        if not p.requires_grad and n in sd:
            del sd[n]

    torch.save({
        'model': sd, 'optimizer': optimizer, 'epoch': epoch, 'step': step,
        'model_selection_metric': metrics, 'warmup_scheduler': scheduler
    }, path)


def get_lrs(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    
    return lrs


@hydra.main(config_path='./configs', config_name='base')
def main(cfg):
    io.mkdir_if_not_exists(cfg.ckpt_dir, recursive=True)
    io.mkdir_if_not_exists(cfg.tb_dir, recursive=True)
    
    if cfg.task.key == 'vl_retrieval':
        cfg.training.batch_size = 32
        cfg.training.num_workers = 8
        cfg.training.num_vis_samples = 5
        cfg.eval.batch_size = 1
        cfg.eval.num_workers = 0
        cfg.eval.num_val_samples = 100
    elif cfg.task.key == 'bongard':
        cfg.training.batch_size = 32
        cfg.training.num_workers = 8
        cfg.eval.batch_size = 32
        cfg.eval.num_workers = 8
    elif cfg.task.key == '3d_reconstruction':
        cfg.training.batch_size = 32
        cfg.eval.batch_size = 50
        cfg.eval.num_val_samples = 100

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
