DATE=$1
BACKBONE=$2
python train.py exp_name=${DATE}-${BACKBONE}-common_sense \
                      task=common_sense \
                      backbone=${BACKBONE} \
                      multiprocessing_distributed=False