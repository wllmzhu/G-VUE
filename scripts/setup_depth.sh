cd ../data/nyuv2/
mkdir -p images annos
cd nyu_depth_v2_labeled

python ../../../preprocessing/nyuv2/extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ../images



