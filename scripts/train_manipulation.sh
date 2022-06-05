DATE=$1
BACKBONE=$2
export CLIPORT_ROOT=/scratch/generalvision/cliport
python train_cliport.py date=${DATE} backbone=${BACKBONE}