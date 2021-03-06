import os
import hydra
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import re
import utils.io as io
import cv2
import numpy as np
from utils.misc import collate_fn
from .base import DATASET
from transforms.bongard_transforms import make_bongard_transforms


BONGARD_TASKS = ['seen_obj_seen_act', 'seen_obj_unseen_act', 'unseen_obj_seen_act', 'unseen_obj_unseen_act']


@DATASET.register()
class BongardHOIDataset(Dataset):
    def __init__(self, info, subset):
        super().__init__()
        self.info = info
        self.subset = subset
        assert self.subset in self.info.subsets, f'subset {self.subset} not in {self.info.subsets}'
        self.transform = make_bongard_transforms(self.subset)
        self._load_dataset()

    def _load_dataset(self):

        if self.subset == 'train':
            anno_filenames = [os.path.join(self.info.anno_dir, 'bongard_hoi_train.json')]
        else: 
            anno_filenames = [os.path.join(self.info.anno_dir, 
                                f'bongard_hoi_{self.subset}_{task}.json') for task in BONGARD_TASKS]

        self.samples = []
        for anno_filename in anno_filenames:
            self.samples.extend(io.load_json_object(anno_filename))

        print(f'load {len(self.samples)} samples in Bongard-HOI {self.subset}')
                
    def __len__(self):
        return len(self.samples)

    def read_image(self, img_name):        
        img = Image.open(os.path.join(self.info.img_dir, img_name)).convert('RGB')
        # Put transform here because later we will process all images together
        img, _ = self.transform(img, None)
        return img

    def __getitem__(self, i):
        sample = self.samples[i]
                
        pos_info_list, neg_info_list, _ = sample

        pos_imgs = []
        for pos_info_i in pos_info_list:
            im_i = self.read_image(pos_info_i['im_path'])
            pos_imgs.append(im_i)

        neg_imgs = []
        for neg_info_i in neg_info_list:
            im_i = self.read_image(neg_info_i['im_path'])
            neg_imgs.append(im_i)
        
        pos_shot_imgs = pos_imgs[:-1]
        pos_query_im = pos_imgs[-1]
        neg_shot_imgs = neg_imgs[:-1]
        neg_query_im = neg_imgs[-1]
        
        pos_shot_imgs = torch.cat(pos_shot_imgs, dim=0)   # [3x6, H, W]
        neg_shot_imgs = torch.cat(neg_shot_imgs, dim=0)   # [3x6, H, W]
        shot_imgs = torch.cat([pos_shot_imgs, neg_shot_imgs], dim=0)   # [3x12, H, W]
        
        input_pos = torch.cat([shot_imgs, pos_query_im], dim=0)   # [3x13, H, W]
        input_neg = torch.cat([shot_imgs, neg_query_im], dim=0)   # [3x13, H, W]

        query_labels = torch.Tensor([0, 1]).long()

        # two multi-channel few-shot samples [2, 3x13, H, W], no text, two labels [2]
        return torch.stack([input_pos, input_neg], dim=0), None, query_labels

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)

@hydra.main(config_path='../configs/task', config_name='bongard.yaml')
def main(cfg):
    dataset = BongardHOIDataset(cfg.dataset.info, 'val')
    dataloader = dataset.get_dataloader(batch_size=2, shuffle=False)
    for data in dataloader:
        imgs, queries, labels = data
        print({
            'image': imgs,
            'query': queries,
            'query label': labels
        })
        break

if __name__ == '__main__':
    main()