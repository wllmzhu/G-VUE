DATE=$1
BACKBONE=$2
python train.py exp_name=${DATE}-${BACKBONE}-vqa \
                      task=vqa \
                      backbone=${BACKBONE} \
                      multiprocessing_distributed=False