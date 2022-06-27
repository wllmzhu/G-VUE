DATE=$1
BACKBONE=$2
python train.py exp_name=${DATE}-${BACKBONE}-depth \
                      task=depth \
                      backbone=${BACKBONE} \
                      multiprocessing_distributed=False