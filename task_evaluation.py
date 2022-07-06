import h5py
import numpy as np
from task_evaluation_lib import evaluate


def evaluate_h5py(pred_h5py, gt_h5py, task):
    print(f'current task: {task}')
    return evaluate(task)(pred_h5py, gt_h5py)


TASKS = [
    'depth', 'camera_relocalization', '3d_reconstruction',
    'vl_retrieval', 'phrase_grounding', 'segmentation',
    'vqa', 'common_sense', 'bongard',
    'navigation', 'manipulation'
]


def main():
    pred_h5py = h5py.File('./submission.h5py', 'r')
    gt_h5py = h5py.File('./gt.h5py', 'r')

    scores = []
    for task in TASKS:
        scores.append(evaluate_h5py(pred_h5py, gt_h5py, task))
    print(f'Overall score of visual representation: {np.mean(scores):.2f}')

    pred_h5py.close()
    gt_h5py.close()
    

if __name__=='__main__':
    main()
