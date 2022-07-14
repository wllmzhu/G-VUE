DATE=$1
BACKBONE=$2
python train.py exp_name=${DATE}-${BACKBONE}-segmentation \
                task=segmentation \
                backbone=${BACKBONE}
