# language-conditioned tasks

# seen for train
task_list_train="
    assembling-kits-seq-seen-colors
    packing-boxes-pairs-seen-colors
    packing-seen-google-objects-seq
    packing-seen-google-objects-group
    put-block-in-bowl-seen-colors
    stack-block-pyramid-seq-seen-colors
    separating-piles-seen-colors
    towers-of-hanoi-seq-seen-colors
"

# unseen for val/test
task_list_test="
    assembling-kits-seq-unseen-colors
    packing-boxes-pairs-unseen-colors
    packing-unseen-google-objects-seq
    packing-unseen-google-objects-group
    put-block-in-bowl-unseen-colors
    stack-block-pyramid-seq-unseen-colors
    separating-piles-unseen-colors
    towers-of-hanoi-seq-unseen-colors
"

# iteratively generate demos, 100 for each task
# train
for task in ${task_list_train}; do
    echo "generating $task for train"
    python ${CLIPORT_ROOT}/cliport/demos.py n=100 \
                                            task=${task} \
                                            mode=train > logs/cliport/train/${task}.log
done

# val
for task in ${task_list_test}; do
    echo "generating ${task} for val"
    python ${CLIPORT_ROOT}/cliport/demos.py n=100 \
                                            task=${task} \
                                            mode=val > logs/cliport/val/${task}.log
done

# test
for task in ${task_list_test}; do
    echo "generating ${task} for test"
    python ${CLIPORT_ROOT}/cliport/demos.py n=100 \
                                            task=${task} \
                                            mode=test > logs/cliport/test/${task}.log
done
