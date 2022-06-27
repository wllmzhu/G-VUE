import hydra
import os

TASKS = ['depth', 'camera_relocalization', '3d_reconstruction',
    'vl_retrieval', 'phrase_grounding', 'segmentation',
    'vqa', 'common_sense', 'bongard', 'navigation', 'manipulation']

@hydra.main(config_path='./configs', config_name='base')
def main(cfg):
    bash_cmd = 'rm submission.h5py'
    os.system(bash_cmd)
    for task in TASKS:
        bash_cmd = f'python3 task_inference.py exp_name={cfg.date}-{cfg.backbone}-{task}' \
                    f'task={task}' \
                    f'backbone={cfg.backbone}' \
                    f'multiprocessing_distributed=False'
        os.system(bash_cmd)
    
