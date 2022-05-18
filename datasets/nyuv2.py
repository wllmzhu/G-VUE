# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from transforms.nyuv2_transforms import make_nyuv2_transforms
from base import DATASET


@DATASET.register()
class NYUv2Dataset(Dataset):
    def __init__(self, info, subset):
        self.info = info
        self.subset = subset
        self.transform = make_nyuv2_transforms(subset)
        self._load_dataset()
    
    def _load_datasets(self):
        if self.subset == 'train':
            with open(self.info.train_index, 'r') as f:
                index_nyuv2 = f.readlines()
        else:
            with open(self.info.test_index, 'r') as f:
                index_nyuv2 = f.readlines()

        self.image_paths = []
        self.depth_paths = []
        for filename_pair in index_nyuv2:
            image_name, depth_name = filename_pair.split()[:2]
            self.image_paths.append(os.path.join(self.info.img_dir, image_name))
            self.depth_paths.append(os.path.join(self.info.img_dir, depth_name))

    def __getitem__(self, i):
        image_path, depth_path = self.image_paths[i], self.depth_paths[i]

        img = Image.open(image_path).convert('RGB')
        depth_gt = Image.open(depth_path)
        
        # To avoid blank boundaries due to pixel registration
        img = img.crop((43, 45, 608, 472))
        depth_gt = depth_gt.crop((43, 45, 608, 472))

        depth_gt = F.to_tensor(depth_gt)
        depth_gt = depth_gt / 1000.0

        target = {'depth': depth_gt}
        img, target = self.transform(img, target)

        # image, text, ground truth
        return img, None, target['depth']
    
    def __len__(self):
        return len(self.image_paths)
