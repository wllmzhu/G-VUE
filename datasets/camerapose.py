import os
import hydra
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import utils.io as io
from transforms.cambridgelandmarks_transforms import make_cambridgelandmarks_transforms
from transforms.seven_scenes_transforms import make_seven_scenes_transforms
from utils.misc import collate_fn
from .base import DATASET
import json
import random
import numpy as np

@DATASET.register()
class CameraPoseDataset(Dataset):
    def __init__(self, info, subset):
        super().__init__()
        self.info = info
        self.subset = subset

        self.dataset_dir = self.info.cambridgelandmarks.dataset_dir if self.info.dataset == 'cambridgelandmarks' else self.info.sevenscenes.dataset_dir
        self.scene = self.info.cambridgelandmarks.scene if self.info.dataset == 'cambridgelandmarks' else self.info.sevenscenes.scene

        self.transforms = make_cambridgelandmarks_transforms() if self.info.dataset == 'cambridgelandmarks' else make_seven_scenes_transforms()

        ## load camera pose dataset
        self._load_data()
    
    def _load_data(self):
        if self.info.dataset == 'cambridgelandmarks':
            phase = 'train' if self.subset == 'train' else 'test'
            txt = os.path.join(self.dataset_dir, self.scene, 'dataset_{}.txt'.format(phase))
        elif self.info.dataset == 'sevenscenes':
            phase = 'train' if self.subset == 'train' else 'val'
            txt = os.path.join(self.dataset_dir, '7Scenes_PoseNet_TrainVal', '{}_{}.txt'.format(self.scene, phase))

        self.samples = [] # [(image, pose), ...]
        with open(txt, 'r') as fp:
            next(fp)  # skip the 3 header lines
            next(fp)
            next(fp)
            for line in fp:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)

                pose = np.array([p0, p1, p2, p3, p4, p5, p6], dtype=np.float32)
                image = os.path.join(
                    self.dataset_dir,
                    self.scene,
                    fname if self.info.dataset == 'cambridgelandmarks' else fname[1:])

                self.samples.append((image, pose))

    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_p, pose = self.samples[i]

        img = Image.open(img_p).convert('RGB')
        img, _ = self.transforms(img, None)

        target = torch.tensor(pose) # [7]
        
        return img, None, target

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)
