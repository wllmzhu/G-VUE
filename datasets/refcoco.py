import os
import hydra
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import utils.io as io
from utils.transforms import make_coco_transforms
from utils.misc import collate_fn
from .base import DATASET


@DATASET.register()
class RefCOCODataset(Dataset):
    def __init__(self, dataset_name, info, subset, task):
        super().__init__()
        self.dataset_name = dataset_name
        self.info = info
        self.subset = subset
        self.task = task
        self.samples = io.load_json_object(
            os.path.join(info.anno_dir, f'{subset}.json')
        )
        print(f'load {len(self.samples)} samples in {self.dataset_name}_{self.subset}')
        self.transform = make_coco_transforms(subset, cautious=True)
    
    def __len__(self):
        return len(self.samples)

    def read_image(self, img_name):
        img = Image.open(os.path.join(self.info.img_dir, img_name)).convert('RGB')
        return img

    def __getitem__(self, i):
        sample = self.samples[i]
        
        img = self.read_image(sample['img_name'])
        query = sample['sentences'][0]['sent']
        bbox = torch.as_tensor(sample['bbox'], dtype=torch.float32)

        target = {
                'query': query,
                'boxes': bbox.view(-1, 4)
            }
        img, target = self.transform(img, target)

        # image, text, ground truth, structure of ground truth, task tag
        return img, target['query'], target['boxes'].squeeze(), 'bbox', self.task

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)


@hydra.main(config_path='../config', config_name='phrase_grounding.yaml')
def main(cfg):
    dataset = RefCOCODataset('refcoco', cfg.dataset.refcoco, 'val', 'phrase_grounding')
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    for data in dataloader:
        imgs, queries, targets, target_type, task_tag = data
        print({
            'image': imgs,
            'text': queries,
            'target': targets,
            'target_type': target_type,
            'task': task_tag
        })
        break


if __name__=='__main__':
    main()
