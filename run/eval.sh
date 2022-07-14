DATE=$1
BACKBONE=$2
TASK=$3

if [ ${TASK} == manipulation ]; then
    export CLIPORT_ROOT=${path_to_cliport}   # customize path to cliport
    python base_scripts/local_eval_cliport.py exp_name=${DATE}-${BACKBONE} backbone=${BACKBONE} \
                                              eval.mode=val eval.checkpoint_type=val_missing \
                                              eval.n_demos=10 &&
    python base_scripts/local_eval_cliport.py exp_name=${DATE}-${BACKBONE} backbone=${BACKBONE} \
                                              eval.mode=test eval.checkpoint_type=test_best
else
    python base_scripts/local_eval.py exp_name=${DATE}-${BACKBONE}-${TASK} \
                                      backbone=${BACKBONE} \
                                      task=${TASK}
fi