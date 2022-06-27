import os
import sys
import hydra
import torch
import numpy as np
import h5py
import json
import yaml
from collections import OrderedDict
import models.nav_decoder.utils as r2r_utils 
from models.nav_decoder.env import R2RBatch
from models.nav_decoder.eval import R2REvaluation
from models.nav_decoder.agent import GVUENavAgent
import cliport.utils as cliport_utils
from cliport.environments.environment import Environment
from torch.utils.data import DataLoader
from models.base import JointModel
from models.manip_decoder.agents import GVUEManipAgent
from datasets.base import create_dataset
from datasets.ravens import RavensDataset
from utils.misc import collate_fn
from task_inference_lib import inference
from hydra.core.hydra_config import HydraConfig

subsets = {
    'depth': ['test'], 'camera_relocalization': ['test'], '3d_reconstruction': ['test'],
    'vl_retrieval': ['test'], 'phrase_grounding': ['val', 'testA', 'testB'], 'segmentation': ['val'],
    'vqa': ['testdev'], 'common_sense': ['val'], 'bongard': ['test'],
    'navigation': ['val_train_seen', 'val_seen', 'val_unseen'],
    'manipulation': [
        'assembling-kits-seq-unseen-colors', 'packing-unseen-google-objects-group',
        'put-block-in-bowl-unseen-colors', 'stack-block-pyramid-seq-unseen-colors',
        'packing-unseen-google-objects-seq', 'packing-boxes-pairs-unseen-colors',
        'separating-piles-unseen-colors', 'towers-of-hanoi-seq-unseen-colors'
    ]
}


def init_model(cfg, device):
    model = JointModel(cfg.model).to(device)
    state_dict = model.state_dict()
    ckpt = torch.load(cfg.eval.ckpt, map_location=device)
    loaded_params = 0
    for k, v in ckpt['model'].items():
        if k in state_dict and state_dict[k].size() == v.size():
            state_dict[k] = v
            loaded_params += 1

    model.load_state_dict(state_dict)
    print(f'loaded {loaded_params} parameters from {cfg.eval.ckpt} for {cfg.task.key} task')
    return model


def generate_camera_pose(cfg, model, h5py_file):
    data_types = ['cambridgelandmarks', 'sevenscenes']
    for data_type in data_types:
        cfg.task.dataset.info.dataset = data_type
        for scene in cfg.task.dataset.info[data_type].all_scenes:
            cfg.task.dataset.info[data_type].scene = scene
            print(f'generating predictions on camera_relocalization task, {data_type} dataset, {scene} scene')
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
                h5py_file = inference('camera_relocalization')(model, dataloader, h5py_file)
    return h5py_file


def generate_nav(cfg, h5py_file):
    sys.path.insert(0, cfg.simulator.build_path)
    import MatterSim  

    val_env_names = ['val_seen', 'val_unseen']
    feat_dict = r2r_utils.read_img_features(cfg.train.data.v_feature, test_only=cfg.test_only)
    # featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
    val_envs = OrderedDict(R2RBatch(cfg, feat_dict, batch_size=cfg.train.setting.batch_size, splits=[split])
                            for split in val_env_names)
    
    train_env = R2RBatch(cfg, feat_dict, batch_size=cfg.train.setting.batch_size, splits=['train'], tokenizer=tok)
    agent = GVUENavAgent(cfg, train_env, "", None, cfg.train.setting.max_action)
    agent.load(cfg.load)

    for env in val_envs.items():
        h5py_file = inference('navigation')(cfg, agent, env, h5py_file)

    return h5py_file


def specify_cliport_ckpt(vcfg):
    result_jsons = [c for c in os.listdir(vcfg['results_path']) if 'results-val' in c]
    if len(result_jsons) > 0:
        result_json = result_jsons[0]
        with open(os.path.join(vcfg['results_path'], result_json), 'r') as f:
            eval_res = json.load(f)
        best_checkpoint = 'last.ckpt'
        best_success = -1.0
        for ckpt, res in eval_res.items():
            if res['average_score'] > best_success:
                best_checkpoint = ckpt
                best_success = res['average_score']
        print(best_checkpoint)
        ckpt = best_checkpoint
    else:
        print("No best val ckpt found. Using last.ckpt")
        ckpt = 'last.ckpt'
    return ckpt


def generate_manip(cfg, h5py_file):
    vcfg = cfg['eval']
    tcfg = cliport_utils.load_hydra_config(vcfg['train_config'])

    env = Environment(
        vcfg['assets_root'],
        disp=vcfg['disp'],
        shared_memory=vcfg['shared_memory'],
        hz=480,
        record_cfg=vcfg['record']
    )

    eval_task = vcfg['task']
    name = '{}-{}-n{}'.format(eval_task, vcfg['agent'], vcfg['n_demos'])
    ckpt = specify_cliport_ckpt(vcfg)
    model_file = os.path.join(vcfg['model_path'], ckpt)
    print(f'evaluating checkpoint: {model_file}')

    for eval_task in subsets['manipulation']:
        ds = RavensDataset(os.path.join(vcfg['data_dir'], f'{eval_task}-test'),
                           tcfg, n_demos=vcfg['n_demos'], augment=False)

        cliport_utils.set_seed(1, torch=True)
        agent = GVUEManipAgent(name, tcfg, None, ds)
        agent.load(model_file)

        h5py_file = inference('manipulation')(env, agent, eval_task, ds, h5py_file, n_demos=vcfg['n_demos'])

    return h5py_file


def generate_group(cfg, h5py_file):
    device = f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu'

    if cfg.task.key == 'camera_relocalization':
        model = init_model(cfg, device)
        h5py_file = generate_camera_pose(cfg, model, h5py_file)
    elif cfg.task.key == 'navigation':
        r2r_cfg = os.path.join(HydraConfig.get().runtime.cwd, 'configs/r2r.yaml')
        with open(r2r_cfg, 'r') as f:
            cfg = yaml.safe_load(f)
        h5py_file = generate_nav(cfg)
    elif cfg.task.key == 'manipulation':
        cliport_cfg = os.path.join(HydraConfig.get().runtime.cwd, 'configs/cliport.yaml')
        with open(cliport_cfg, 'r') as f:
            cfg = yaml.safe_load(f)
        h5py_file = generate_manip(cfg)
    else:
        model = init_model(cfg, device)
        print(f'generating predictions on {cfg.task.key} task, {cfg.task.dataset.key} dataset')
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
            h5py_file = inference(cfg.task.key)(model, dataloader, h5py_file)
    
    return h5py_file


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
    
    h5py_file = h5py.File('preds_to_submit.h5py', 'w')
    h5py_file = generate_group(cfg, h5py_file)
    h5py_file.close()
    

if __name__=='__main__':
    main()
