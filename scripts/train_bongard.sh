DATE=$1
BACKBONE=$2
python train.py exp_name=${DATE}-${BACKBONE}-bongard \
                      task=bongard \
                      backbone=${BACKBONE} \
                      multiprocessing_distributed=False