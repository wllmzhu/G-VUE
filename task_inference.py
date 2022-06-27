import torch
import torch.nn.functional as F
import numpy as np
import h5py
from math import ceil
from tqdm import tqdm
from cliport import tasks


@torch.no_grad()
def InferDepth(model, dataloader, h5py_file):
    grp = h5py_file.create_group('depth')
    model.eval()
    min_depth, max_depth = 1e-3, 10
    all_preds = []

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        outputs = model(imgs, txts=None)
        preds = (outputs.sigmoid().squeeze() * max_depth).clip(min_depth, max_depth)
        all_preds.append(preds.detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    grp.create_dataset(f'{dataloader.dataset.subset}', data=all_preds)
    return h5py_file


@torch.no_grad()
def InferCameraRelocalization(model, dataloader, h5py_file):
    grp = h5py_file.create_group('camera_relocalization') if 'camera_relocalization' not in h5py_file.keys() \
                                                          else h5py_file['camera_relocalization']
    model.eval()
    all_preds = []

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        outputs = model(imgs)
        all_preds.append(outputs.detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    grp.create_dataset(f'{dataloader.dataset.scene}', data=all_preds)
    return h5py_file


@torch.no_grad()
def InferRec(model, dataloader, h5py_file):
    grp = h5py_file.create_group('3d_reconstruction')
    model.eval()
    all_preds = []

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        img_logits = model(imgs)
        outputs = model.decoder.inference(img_logits)
        pred_vox = outputs < 0
        all_preds.append(pred_vox.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    grp.create_dataset(f'{dataloader.dataset.subset}', data=all_preds)
    return h5py_file


@torch.no_grad()
def InferRetrieval(model, dataloader, h5py_file):
    grp = h5py_file.create_group('vl_retrieval')
    model.eval()

    txts = dataloader.dataset.texts
    step_size = 1000

    all_preds = []

    for data in tqdm(dataloader):
        imgs, _, targets = data
        B = len(targets)
        
        img2txt_scores = []
        num_step = ceil(len(txts)/step_size)
        for i in range(num_step):
            txt_batch = [txts[i*step_size:(i+1)*step_size]] * B
            outputs = model(imgs, txt_batch, task='vl_retrieval')
            img2txt_scores.append(outputs.view(B, -1))

        img2txt_scores = torch.cat(img2txt_scores, dim=-1)
        # [B, len(txts)]
        _, preds = torch.sort(img2txt_scores, dim=-1, descending=True)
        all_preds.append(preds.detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    grp.create_dataset(f'{dataloader.dataset.subset}', data=all_preds)
    return h5py_file


@torch.no_grad()
def InferBbox(model, dataloader, h5py_file):
    grp = h5py_file.create_group('phrase_grounding') if 'phrase_grounding' not in h5py_file.keys() \
                                                     else h5py_file['phrase_grounding']
    model.eval()
    all_preds = []

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        outputs = model(imgs, txts).sigmoid()
        preds = torch.stack([
            outputs[:, 0]-outputs[:, 2]/2,
            outputs[:, 1]-outputs[:, 3]/2,
            outputs[:, 0]+outputs[:, 2]/2,
            outputs[:, 1]+outputs[:, 3]/2
        ], dim=1)
        all_preds.append(preds.detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    grp.create_dataset(f'{dataloader.dataset.subset}', data=all_preds)
    return h5py_file


@torch.no_grad()
def InferSeg(model, dataloader, h5py_file):
    grp = h5py_file.create_group('segmentation')
    model.eval()
    all_preds = []

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        outputs = model(imgs, txts=None)
        all_preds.append(outputs.argmax(1).detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    grp.create_dataset(f'{dataloader.dataset.subset}', data=all_preds)
    return h5py_file


@torch.no_grad()
def InferQA(model, dataloader, h5py_file):
    grp = h5py_file.create_group('vqa')
    model.eval()
    all_preds = []

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        outputs = model(imgs, txts)
        preds = torch.topk(outputs, k=1, dim=-1).indices.squeeze()
        all_preds.append(preds.detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    grp.create_dataset(f'{dataloader.dataset.subset}', data=all_preds)
    return h5py_file


@torch.no_grad()
def InferVCR(model, dataloader, h5py_file):
    grp = h5py_file.create_group('common_sense')
    model.eval()
    all_preds = []

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        B = len(targets)
        outputs = model(imgs, txts, task='common_sense')
        # [4B, 1] -> [B, 4]
        outputs = outputs.view(B, -1)
        preds = torch.topk(outputs, k=1, dim=-1).indices.squeeze()   # [B]
        all_preds.append(preds.detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    grp.create_dataset(f'{dataloader.dataset.subset}', data=all_preds)
    return h5py_file


@torch.no_grad()
def InferBongard(model, dataloader, h5py_file):
    grp = h5py_file.create_group('bongard')
    model.eval()
    all_preds = []

    for data in tqdm(dataloader):
        imgs, txts, targets = data
        outputs = model(imgs, txts, task='bongard')
        # [2B, 2] -> [2B]
        preds = torch.topk(outputs, k=1, dim=-1).indices.squeeze()
        all_preds.append(preds.detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    grp.create_dataset(f'{dataloader.dataset.subset}', data=all_preds)
    return h5py_file


@torch.no_grad()
def InferNav(cfg, agent, env, h5py_file):
    grp = h5py_file.create_group('navigation')
    agent.vln_bert.eval()
    agent.critic.eval()

    agent.env = env
    agent.test(use_dropout=False, feedback='argmax', iters=None)
    result = agent.get_results()

    grp.create_dataset(f'{env.name}', data=result)
    return h5py_file


@torch.no_grad()
def InferManip(env, agent, eval_task, dataset, h5py_file, n_demos=100):
    grp = h5py_file.create_group('manipulation') if 'manipulation' not in h5py_file.keys() \
                                                 else h5py_file['manipulation']
    task = tasks.names[eval_task]()
    total_rewards = []
    total_actions = []
    print(f'Evaluating on {eval_task}')
    for i in range(0, n_demos):
        print(f'Test: {i + 1}/{n_demos}')
        episode, seed = dataset.load(i)
        goal = episode[-1]
        total_reward = 0
        actions = []
        np.random.seed(seed)

        task.mode = 'test'
        env.seed(seed)
        env.set_task(task)
        obs = env.reset()
        info = env.info

        for _ in range(task.max_steps):
            act = agent.act(obs, info, goal)
            actions.append(act)
            obs, reward, done, info = env.step(act)
            total_reward += reward
            if done:
                break

        total_rewards.append(total_reward)
        total_actions.append(actions)
    
    grp.create_dataset(f'{eval_task}', data={'rewards': total_rewards, 'actions': total_actions})
    return h5py_file
  

task_infer_dict = {
    'depth': InferDepth, 'camera_relocalization': InferCameraRelocalization, '3d_reconstruction': InferRec,
    'vl_retrieval': InferRetrieval, 'phrase_grounding': InferBbox, 'segmentation': InferSeg,
    'vqa': InferQA, 'common_sense': InferVCR, 'bongard': InferBongard,
    'navigation': InferNav, 'manipulation': InferManip
}


def inference(task):
    return task_infer_dict[task]
