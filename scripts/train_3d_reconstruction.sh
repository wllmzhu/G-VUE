DATE=$1
BACKBONE=$2
python train.py exp_name=${DATE}-${BACKBONE}-3d_reconstruction \
                      task=3d_reconstruction \
                      backbone=${BACKBONE} \
                      multiprocessing_distributed=False