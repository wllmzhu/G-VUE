import os
import hydra
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import utils.io as io
from transforms.rec_transforms import make_rec_transforms
from utils.misc import collate_fn
from .base import DATASET
import json
import random
import numpy as np


## ShapeNetCore.v1, 13 categories
## Rendering images come from DISN, https://github.com/Xharlie/DISN
## Models are pre-processed by autosdf, https://github.com/yccyenchicheng/AutoSDF
## Train-test split follows DISN
## Data contains images, codeidx, volumetric tsdf (trunc_thres=0.2)
@DATASET.register()
class ShapeNetDataset(Dataset):
    def __init__(self, info, subset):
        super().__init__()
        self.info = info
        self.subset = subset

        with open(self.info.info_file, 'r') as fp:
            data = json.load(fp)
            self.cats_id = data['cats']

            if self.info.cat == 'all':
                self.all_cats = data['all_cats']
            else:
                self.all_cats = [self.info.cat]

        self.transforms = make_rec_transforms(self.subset)

        ## load shapenet data
        self._load_data()
    
    def _load_data(self):
        phase = 'train' if self.subset == 'train' else 'test'

        self.samples = [] # [(image, code index, volumetric tsdf), ...]
        for c in self.all_cats:
            synset = self.cats_id[c]
            
            with open(os.path.join(self.info.split_dir, synset+'_'+phase+'.lst'), 'r') as f:
                sample_list_s = []
                for l in f.readlines():
                    model_id = l.rstrip('\n')
                    
                    image_path = os.path.join(self.info.img_dir, synset, model_id, 'rendering')
                    codeix_path = os.path.join(self.info.anno_dir, "pvqvae-snet-{}-T0.2/codeix".format(self.info.cat), synset, model_id, 'codeix.npy')
                    x_path = os.path.join(self.info.anno_dir, "pvqvae-snet-{}-T0.2/x".format(self.info.cat), synset, model_id, 'x.npy')

                    sample_list_s.append((image_path, codeix_path, x_path))

                self.samples += sample_list_s
        
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_p, idx_p, sdf_p = self.samples[i]

        img_p = os.path.join(img_p, '{:0>2d}.png'.format(random.randint(0, 23))) # randomly select one image from 24 images of different views
        img = Image.open(img_p).convert('RGB')
        img, _ = self.transforms(img, None)
        idx = torch.tensor(np.load(idx_p)) # [8, 8, 8]
        sdf = torch.tensor(np.load(sdf_p)) # [1, 64, 64, 64]

        # image, text, ground truth (idx, sdf)
        return img, None, (idx, sdf)

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
