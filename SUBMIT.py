import os

TASKS = [
    'depth', 'camera_relocalization', '3d_reconstruction',
    'vl_retrieval', 'phrase_grounding', 'segmentation',
    'vqa', 'common_sense', 'bongard',
    'navigation', 'manipulation'
]

#---------------------------------------------------
# Modify to customize submission

EXPS = {
    'depth': None, 'camera': None, '3d_reconstruction': None,
    'vl_retrieval': None, 'phrase_grounding': None, 'segmentation': None,
    'vqa': None, 'common_sense': None, 'bongard': None,
    'navigation': None, 'manipulation': None
}

BACKBONE = None

#---------------------------------------------------


def main():
    print('Generate h5py for submission')

    bash_cmd = 'rm submission.h5py'
    os.system(bash_cmd)

    for task in TASKS:
        exp_name = EXPS[task]
        bash_cmd = f'python task_inference.py exp_name={exp_name} task={task} backbone={BACKBONE}'
        os.system(bash_cmd)


if __name__ == '__main__':
    main()
