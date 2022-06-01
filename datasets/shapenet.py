import os
import hydra
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import utils.io as io
from transforms.refcoco_transforms import make_coco_transforms
from utils.misc import collate_fn
from .base import DATASET


@DATASET.register()
class ShapeNetDataset(Dataset):
    def __init__(self, info, subset):
        super().__init__()
        self.info = info
        self.subset = subset

        ## load shapenet data
        
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        img, idx, vox = sample
        
        # image, text, ground truth (idx, vox)
        return img, None, (idx, vox)

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)


@hydra.main(config_path='../config', config_name='3d_reconstruction.yaml')
def main(cfg):
    dataset = ShapeNetDataset(cfg.dataset.info, 'val')
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
