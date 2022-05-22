import os
import pickle as pkl
import numpy as np
import hydra
from PIL import Image
import torch
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
        self._load_class_dict()
    
    def _load_dataset(self):
        with open(self.info.index_file, 'rb') as f:
            self.index_ade20k = pkl.load(f)

        self.train_idx = []
        self.val_idx = []
        for idx, filename in enumerate(self.index_ade20k['filename']):
            if 'train' in filename:
                self.train_idx.append(idx)
            elif 'val' in filename:
                self.val_idx.append(idx)
        # Data also contains "frame" data like ADE_frame_00000001.jpg, ignored

        print(f'load {self.__len__()} samples in ADE20k {self.subset}')
    
    def _load_class_dict(self):
        with open(self.info.class_dict, 'r') as f:
            idx2cat = f.readlines()
        self.class_dict = {}
        for line in idx2cat:
            idx, cat = line.split()
            idx, cat = int(idx), int(cat)
            self.class_dict.update({cat: idx})
        self.class_dict.update({0: 0})
        print('constructed mapping between category and class index')

    def __len__(self):
        return len(self.train_idx) if self.subset == 'train' else len(self.val_idx)

    def __getitem__(self, i):
        if self.subset == 'train':
            i = self.train_idx[i]
        else:
            i = self.val_idx[i]

        filename = os.path.join(self.info.top_dir, self.index_ade20k['folder'][i], self.index_ade20k['filename'][i])
        fileseg = filename.replace('.jpg', '_seg.png')

        img = Image.open(filename)

        with Image.open(fileseg) as io:
            seg = np.array(io)
        r = seg[:,:,0]
        g = seg[:,:,1]
        category = (r/10).astype(np.int32)*256+(g.astype(np.int32))
        category = F.to_tensor(category)
        category.apply_(self.class_dict.get)

        target = {'depth': category}
        img, target = self.transform(img, target)

        # image, segmentation
        return img, None, target['depth']

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)


@hydra.main(config_path='../configs/task', config_name='segmentation.yaml')
def main(cfg):
    dataset = ADE20kDataset(cfg.dataset.info, 'val')
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    for data in dataloader:
        imgs, targets = data
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
