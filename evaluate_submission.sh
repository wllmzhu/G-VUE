DATE=$1
BACKBONE=$2

if [ -z "$1" ]
then
echo "No DATE supplied"
exit 1
fi

if [ -z "$2" ]
then
echo "No BACKBONE supplied"
exit 1
fi

TASKS="depth camera_relocalization 3d_reconstruction
        vl_retrieval phrase_grounding segmentation vqa
        common_sense bongard navigation manipulation"

for task in $TASKS; 
do 
    python evaluate_submission.py exp_name=${DATE}-${BACKBONE}-${task} \
                                    task=${task} \
                                    backbone=${BACKBONE} \
                                    multiprocessing_distributed=False
done