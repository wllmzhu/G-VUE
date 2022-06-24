# DATE=$1
# BACKBONE=$2
python test.py --multirun \
                task=depth,vqa \
                # exp_name=${DATE}-${BACKBONE}-depth \
                # backbone=${BACKBONE} \
                # multiprocessing_distributed=False