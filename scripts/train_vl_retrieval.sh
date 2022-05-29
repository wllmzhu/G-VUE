DATE=$1
BACKBONE=$2
python train_distr.py exp_name=${DATE}-${BACKBONE}-vl_retrieval \
                      task=vl_retrieval \
                      backbone=${BACKBONE} \
                      multiprocessing_distributed=False