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

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from transforms.nyuv2_transforms import rotate_image, random_crop, train_preprocess, ToTensor
from transforms.nyuv2_transforms import make_nyuv2_transforms
            
class NYUv2Dataset(Dataset):
    def __init__(self, info, subset):
        self.info = info
        self.subsets = ['train', 'test']
        self.subset = subset
        assert self.subset in self.subsets, f'subset {self.subset} not in {self.subsets}'
        self.transform = make_nyuv2_transforms(subset)
        self._load_dataset()
    
    def _load_datasets(self):
        with open(self.info.index_file, 'r') as f:
            index_nyuv2 = f.readlines()

        self.image_paths = []
        self.depth_paths = []
        for filename_pair in index_nyuv2:
            self.image_paths.append(os.path.join(self.args.data_path, "./" + filename_pair.split()[0]))
            self.depth_paths.append(os.path.join(self.args.gt_path, "./" + filename_pair.split()[1]))
        

    def __getitem__(self, i):
        image_path, depth_path = self.image_paths[i], self.depth_paths[i]

        img = Image.open(image_path)
        depth_gt = Image.open(depth_path)
        
        # To avoid blank boundaries due to pixel registration
        depth_gt = depth_gt.crop((43, 45, 608, 472))
        img = img.crop((43, 45, 608, 472))

        img = np.asarray(img, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)
        depth_gt = depth_gt / 1000.0

        target = {
                'depth': depth_gt,
            }
        img, target = self.transform(img, target)

        # image, text, ground truth
        return img, target['depth']
    

    def __len__(self):
        return len(self.image_paths)

