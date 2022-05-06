import os
import hydra
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import utils.io as io
from utils.nyuv2_transforms import make_nyuv2_transforms
from utils.misc import collate_fn


class NYUv2Dataset(Dataset):
    def __init__(self, dataset_name, info, subset, task):
        super().__init__()
        self.dataset_name = dataset_name
        self.info = info
        self.task = task
        self.file = io.load_hdf5_object(info['hdf5_path'])
        self.color_maps = self.file['images']
        self.depth_maps = self.file['depths']
        print(f'load {len(self.images)} samples in {self.dataset_name}')
        self.transform = make_nyuv2_transforms(subset)
    
    def __len__(self):
        return len(self.samples)

    def rotate_image(image):
        return image.rotate(-90, expand=True)

    def __getitem__(self, i):
        color_map = self.color_maps[i]
        color_map = np.moveaxis(color_map, 0, -1)
        color_image = Image.fromarray(color_map, mode='RGB')
        color_image = self.rotate_image(color_image)

        depth_map = self.depth_maps[i]
        depth_image = Image.fromarray(depth_map, mode='F')
        depth_image = self.rotate_image(depth_image)

        target = {
                'mask': depth_image,
            }
        img, target = self.transform(color_image, target)

        # image, text, ground truth, structure of ground truth, task tag
        return img, target['mask'], target['boxes'].squeeze(), 'image', self.task

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)


@hydra.main(config_path='../config', config_name='depth.yaml')
def main(cfg):
    dataset = NYUv2Dataset('nyuv2', cfg.dataset.nyuv2, 'val', 'depth')
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    for data in dataloader:
        imgs, targets, target_type, task_tag = data
        print({
            'image': imgs,
            'target': targets,
            'target_type': target_type,
            'task': task_tag
        })
        break


if __name__=='__main__':
    main()
