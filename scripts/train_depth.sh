DATE=$1
BACKBONE=$2
python train_distr.py exp_name=${DATE}-${BACKBONE}-depth \
                      task=depth \
                      backbone=${BACKBONE} \
                      multiprocessing_distributed=False