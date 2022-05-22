import os
import pickle as pkl
import numpy as np
import hydra
from PIL import Image
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
import utils.io as io
from transforms.ade20k_transforms import make_ade20k_transforms
from utils.misc import collate_fn
from .base import DATASET


@DATASET.register()
class ADE20kDataset(Dataset):
    def __init__(self, info, subset):
        super().__init__()
        self.info = info
        self.subset = subset
        assert self.subset in self.info.subsets, f'subset {self.subset} not in {self.info.subsets}'
        self.transform = make_ade20k_transforms(subset)
        self._load_dataset()
    
    def _load_dataset(self):
        self.samples = [x.rstrip() for x in open(os.path.join(self.info.index_dir, f'ADE20K_object150_{self.subset}.txt'), 'r')]
        print(f'load {len(self.samples)} samples in ADE20K (MIT Scene Parsing Benchmark) {self.subset}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        filename = self.samples[i]
        img_filename = os.path.join(self.info.img_dir, filename)
        seg_filename = os.path.join(self.info.anno_dir, filename)
        seg_filename = seg_filename.replace('.jpg', '.png')

        img = Image.open(img_filename).convert('RGB')
        seg = Image.open(seg_filename)

        seg = F.to_tensor(seg)
        # note: here uint8 is converted to [0, 1] automatically

        target = {'depth': seg}

        img, target = self.transform(img, target)
        # [0, 1] back to [0, 255], loss computing requires dtype long
        seg = (target['depth'] * 255).squeeze().to(torch.long)
        # image, segmentation
        return img, None, seg

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)


@hydra.main(config_path='../configs/task', config_name='segmentation.yaml')
def main(cfg):
    dataset = ADE20kDataset(cfg.dataset.info, 'val')
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    for data in dataloader:
        print(len(data))
        imgs, txt, targets = data
        print({
            'image': imgs[0],
            'target': targets[0],
        })
        
        import matplotlib.pyplot as plt
        plt.imshow(targets[0].permute(1, 2, 0))
        plt.show()

        break


if __name__=='__main__':
    main()
