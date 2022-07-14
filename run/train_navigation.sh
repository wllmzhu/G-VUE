DATE=$1
BACKBONE=$2
python setup/preprocess/r2r/generate_v_features.py backbone=${BACKBONE} \
                                                   exp_name=${DATE}-${BACKBONE}-navigation
python base_scripts/train_r2r.py backbone=${BACKBONE} \
                                 exp_name=${DATE}-${BACKBONE}-navigation
