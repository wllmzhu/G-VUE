DATE=$1
BACKBONE=$2
python train_distr.py exp_name=${DATE}-${BACKBONE}-camera_relocalization \
                      task=camera_relocalization \
                      backbone=${BACKBONE} \
                      multiprocessing_distributed=False