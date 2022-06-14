import os
import hydra
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import utils.io as io
from transforms.seven_scenes_transforms import make_seven_scenes_transforms
from utils.misc import collate_fn
from .base import DATASET
import json
import random
import numpy as np

@DATASET.register()
class SevenScenesDataset(Dataset):
    def __init__(self, info, subset):
        super().__init__()
        self.info = info
        self.subset = subset

        self.transforms = make_seven_scenes_transforms()

        ## load kingscollege data
        self._load_data()
    
    def _load_data(self):
        phase = 'train' if self.subset == 'train' else 'val'

        self.samples = [] # [(image, pose), ...]
        with open(os.path.join(self.info.dataset_dir, '7Scenes_PoseNet_TrainVal', '{}_{}.txt'.format(self.info.scene, phase)), 'r') as fp:
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
                image = os.path.join(self.info.dataset_dir, self.info.scene, fname[1:])

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


@hydra.main(config_path='../config', config_name='camera_relocalization.yaml')
def main(cfg):
    dataset = SevenScenesDataset(cfg.dataset.info, 'val')
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    for data in dataloader:
        imgs, txts, targets = data
        print({
            'image': imgs,
            'text': txts,
            'target': targets
        })
        break


if __name__=='__main__':
    main()
