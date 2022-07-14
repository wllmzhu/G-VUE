DATE=$1
BACKBONE=$2
python base_scripts/train.py exp_name=${DATE}-${BACKBONE}-3d_reconstruction \
                             task=3d_reconstruction \
                             backbone=${BACKBONE}
