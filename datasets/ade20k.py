import os
import pickle as pkl
import numpy as np
import hydra
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import utils.io as io
from transforms.ade20k_transforms import make_ade20k_transforms
from utils.misc import collate_fn
from base import DATASET


@DATASET.register()
class ADE20kDataset(Dataset):
    def __init__(self, info, subset):
        super().__init__()
        self.info = info
        self.subset = subset
        self.transform = make_ade20k_transforms()
        self._load_dataset()
    
    def _load_dataset(self):
        with open(self.info.index_file, 'rb') as f:
            self.index_ade20k = pkl.load(f)

        print(f'load {len(self.index_ade20k["filename"])} samples in ADE20k {self.subset}')

    def __len__(self):
        return len(self.index_ade20k["filename"])

    def __getitem__(self, i):
        filename = self.index_ade20k['filename'][i]  
        fileseg = filename.replace('.jpg', '_seg.png')

        img = Image.open(filename)

        with Image.open(fileseg) as io:
            seg = np.array(io)
        r = seg[:,:,0]
        g = seg[:,:,1]
        category = (r/10).astype(np.int32)*256+(g.astype(np.int32))

        target = {
                'depth': category}
        img, target = self.transform(img, target)

        # image, segmentation
        return img, target['depth']

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)


@hydra.main(config_path='../configs/task', config_name='segmentation.yaml')
def main(cfg):
    dataset = ADE20kDataset(cfg.dataset.info, 'val')
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    for data in dataloader:
        imgs, targets = data
        print({
            'image': imgs,
            'target': targets,
        })
        break


if __name__=='__main__':
    main()
