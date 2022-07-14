DATE=$1
BACKBONE=$2
python preprocess/r2r/generate_v_features.py backbone=${BACKBONE} \
                                             exp_name=${DATE}-${BACKBONE}-navigation
python train_r2r.py backbone=${BACKBONE} \
                    exp_name=${DATE}-${BACKBONE}-navigation
