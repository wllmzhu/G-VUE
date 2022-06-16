import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from transforms.nyuv2_transforms import make_nyuv2_transforms
from .base import DATASET


@DATASET.register()
class NYUv2Dataset(Dataset):
    def __init__(self, info, subset):
        self.info = info
        self.subset = subset
        self.transform = make_nyuv2_transforms(subset)
        self._load_dataset()
    
    def _load_dataset(self):
        if self.subset == 'train':
            img_dir = self.info.train_img_dir
            with open(self.info.train_index, 'r') as f:
                index_nyuv2 = f.readlines()
        else:
            img_dir = self.info.test_img_dir
            with open(self.info.test_index, 'r') as f:
                index_nyuv2 = f.readlines()

        self.image_paths = []
        self.depth_paths = []
        for filename_pair in index_nyuv2:
            image_name, depth_name = filename_pair.split()[:2]
            self.image_paths.append(os.path.join(img_dir, image_name))
            self.depth_paths.append(os.path.join(img_dir, depth_name))

    def __getitem__(self, i):
        image_path, depth_path = self.image_paths[i], self.depth_paths[i]

        img = Image.open(image_path).convert('RGB')
        depth_gt = Image.open(depth_path)
        
        # To avoid blank boundaries due to pixel registration
        img = img.crop((43, 45, 608, 472))
        depth_gt = depth_gt.crop((43, 45, 608, 472))

        depth_gt = F.to_tensor(depth_gt)
        depth_gt = depth_gt / 1000.0

        target = {'depth': depth_gt}
        img, target = self.transform(img, target)

        # image, text, ground truth
        return img, None, target['depth']
    
    def __len__(self):
        return len(self.image_paths)
