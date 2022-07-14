DATE=$1
BACKBONE=$2
python base_scripts/train.py exp_name=${DATE}-${BACKBONE}-vl_retrieval \
                             task=vl_retrieval \
                             backbone=${BACKBONE}
