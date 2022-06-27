DATE=$1
BACKBONE=$2
python train.py exp_name=${DATE}-${BACKBONE}-vl_retrieval \
                      task=vl_retrieval \
                      backbone=${BACKBONE} \
                      multiprocessing_distributed=False