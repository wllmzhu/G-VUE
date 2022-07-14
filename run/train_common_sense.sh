DATE=$1
BACKBONE=$2
python base_scripts/train.py exp_name=${DATE}-${BACKBONE}-common_sense \
                             task=common_sense \
                             backbone=${BACKBONE}
