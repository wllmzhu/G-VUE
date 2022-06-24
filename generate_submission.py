import os
import hydra
import torch
import numpy as np
import h5py
import json
import cliport.utils as utils
from torch.utils.data import DataLoader
from models.base import JointModel
from datasets.base import create_dataset
from datasets.ravens import RavensDataset
from utils.misc import collate_fn
from inference_all_tasks import inference

subsets = {
    'depth': ['test'], 'camera_relocalization': ['test'], '3d_reconstruction': ['test'],
    'vl_retrieval': ['test'], 'phrase_grounding': ['val', 'testA', 'testB'], 'segmentation': ['val'],
    'vqa': ['testdev'], 'common_sense': ['val'], 'bongard': ['test']
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


def specify_cliport_ckpt(vcfg):
    result_jsons = [c for c in os.listdir(vcfg.results_path) if "results-val" in c]
    if len(result_jsons) > 0:
        result_json = result_jsons[0]
        with open(os.path.join(vcfg.results_path, result_json), 'r') as f:
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
    for eval_task in UNSEEN_TASKS:
        ds = RavensDataset(os.path.join(vcfg.data_dir, f'{eval_task}-test_best'),
                           tcfg, n_demos=vcfg.n_demos, augment=False)

        results = []
        mean_reward = 0.0

        # Initialize agent.
        utils.set_seed(1, torch=True)
        agent = GVUEAgent(name, tcfg, None, ds)

        # Load checkpoint
        agent.load(model_file)
        print(f"Loaded: {model_file}")

        record = vcfg.record.save_video
        n_demos = vcfg.n_demos

        # Run testing and save total rewards with last transition info.
        for i in range(0, n_demos):
            print(f'Test: {i + 1}/{n_demos}')
            episode, seed = ds.load(i)
            goal = episode[-1]
            total_reward = 0
            np.random.seed(seed)

            # set task
            if 'multi' in dataset_type:
                task_name = ds.get_curr_task()
                task = tasks.names[task_name]()
            else:
                task_name = eval_task
                task = tasks.names[task_name]()
            print(f'Evaluating on {task_name}')

            task.mode = mode
            env.seed(seed)
            env.set_task(task)
            obs = env.reset()
            info = env.info
            reward = 0

            # Start recording video (NOTE: super slow)
            if record:
                video_name = f'{task_name}-{i+1:06d}'
                if 'multi' in vcfg.model_task:
                    video_name = f"{vcfg.model_task}-{video_name}"
                env.start_rec(video_name)

            for _ in range(task.max_steps):
                act = agent.act(obs, info, goal)
                lang_goal = info['lang_goal']
                print(f'Lang Goal: {lang_goal}')
                print(f'Action: {act}')
                obs, reward, done, info = env.step(act)
                total_reward += reward
                print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')
                if done:
                    break

            results.append((total_reward, info))
            mean_reward = np.mean([r for r, i in results])
            print(f'Mean: {mean_reward} | Task: {task_name} | Ckpt: {ckpt}')

            all_results[ckpt][eval_task] = {
                'episodes': results,
                'mean_reward': mean_reward,
            }

    # average scores on various tasks
    average_scores = [all_results[ckpt][eval_task]['mean_reward'] for eval_task in UNSEEN_TASKS]
    average_scores = np.mean(average_scores)
    print(f'average scores on tasks for {ckpt}: {average_scores}')
    all_results[ckpt].update({'average_score': average_scores})


def generate_group(cfg, h5py_file):
    device = f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu'

    if cfg.task.key == 'camera_relocalization':
        model = init_model(cfg, device)
        generate_camera_pose(cfg, model, h5py_file)
    elif cfg.task.key == 'navigation':
        raise NotImplementedError
    elif cfg.task.key == 'manipulation':
        generate_manip(cfg, model, )
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
    
    h5py_file = h5py.File('preds_to_submit.h5py', 'w')
    generate_group(cfg, h5py_file)
    

if __name__=='__main__':
    main()
