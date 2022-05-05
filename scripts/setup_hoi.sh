mkdir -p ../data/hoi/hico_processed

python -m preprocessing.hicodet.mat_to_json
python -m preprocessing.hicodet.hoi_cls_count
python -m preprocessing.hicodet.split_ids