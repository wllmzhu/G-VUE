DATE=$1
BACKBONE=$2
python train.py exp_name=${DATE}-${BACKBONE}-phrase_grounding \
                      task=phrase_grounding \
                      backbone=${BACKBONE} \
                      multiprocessing_distributed=False