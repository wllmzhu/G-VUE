DATE=$1
BACKBONE=$2
<<<<<<< HEAD
python preprocess/r2r/generate_v_features.py date=${DATE} \
                                                backbone=${BACKBONE} \
                                                exp_name=${DATE}-${BACKBONE}-navigation
python train_r2r.py date=${DATE} \
                    backbone=${BACKBONE} \
                    exp_name=${DATE}-${BACKBONE}-navigation
=======

python preprocess/r2r/generate_v_features.py date=$DATE \
                                             backbone=${BACKBONE} \
                                             exp_name=${DATE}-${BACKBONE}-navigation

python train_r2r.py date=$DATE \
                    backbone=${BACKBONE} 
                    exp_name=${DATE}-${BACKBONE}-navigation
>>>>>>> 0b03192... finish submission
