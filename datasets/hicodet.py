# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from torchvision.datasets.vision import VisionDataset
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from utils.hicodet_transforms import parse_one_gt_line, make_hicodet_transforms

class HICODET(VisionDataset):
    """
        You are supposed to make a soft link named 'images' in 'data/hico/' to refer to your HICO-DET images' path.
        E.g. ln -s /path-to-your-hico-det-dataset/hico_20160224_det/images images
    """
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, image_set='train'):
        assert image_set in ['train', 'test'], image_set
        self.image_set = image_set
        super(HICODET, self).__init__(root, transforms, transform, target_transform)
        annotations = [parse_one_gt_line(l.strip()) for l in open(annFile, 'r').readlines()]
        if self.image_set in ['train']:
            self.annotations = [a for a in annotations if len(a['annotations']['action_labels']) > 0]
        else:
            self.annotations = annotations
        self.transforms = transforms

    def __getitem__(self, index):
        ann = self.annotations[index]
        img_name = ann['image_id']
        target = ann['annotations']
        if 'train2015' in img_name:
            img_path = '../data/hicodet/images/train2015/%s' % img_name
        elif 'test2015' in img_name:
            img_path = '../data/hicodet/images/test2015/%s' % img_name
        else:  # For single image visualization.
            raise NotImplementedError()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target, self.image_set)
        return img, target

    def __len__(self):
        return len(self.annotations)


def build(image_set, test_scale=-1):
    assert image_set in ['train', 'test'], image_set
    if image_set == 'train':
        annotation_file = '../data/hicodet/hico_trainval_remake.odgt'
    else:
        annotation_file = '../data/hicodet/hico_test_remake.odgt'

    dataset = HICODET(root='../data/hicodet', annFile=annotation_file,
                           transforms=make_hicodet_transforms(image_set, test_scale), image_set=image_set)
    return dataset


def main():
    dataset = build("test")
    print(dataset[0])

if __name__ == '__main__':
    main()