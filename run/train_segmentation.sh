DATE=$1
BACKBONE=$2
python base_scripts/train.py exp_name=${DATE}-${BACKBONE}-segmentation \
                             task=segmentation \
                             backbone=${BACKBONE}
