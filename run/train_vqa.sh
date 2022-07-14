DATE=$1
BACKBONE=$2
python base_scripts/train.py exp_name=${DATE}-${BACKBONE}-vqa \
                             task=vqa \
                             backbone=${BACKBONE}
