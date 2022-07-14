DATE=$1
BACKBONE=$2
python train.py exp_name=${DATE}-${BACKBONE}-camera_relocalization \
                task=camera_relocalization \
                backbone=${BACKBONE}
