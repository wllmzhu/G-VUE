import os
import hydra
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import re
import utils.io as io
from utils.misc import collate_fn
from .base import DATASET
from transforms.flickr30k_transforms import make_flickr30k_transforms


@DATASET.register()
class Flickr30kDataset(Dataset):
    def __init__(self, info, subset):
        super().__init__()
        self.info = info
        self.subset = subset
        assert self.subset in self.info.subsets, f'subset {self.subset} not in {self.info.subsets}'
        self.transform = make_flickr30k_transforms()
        self._load_dataset()

    def _load_dataset(self):
        self.samples = io.load_json_object(
            os.path.join(self.info.anno_dir, f'flickr30k_{self.subset}.json')
        )
        print(f'load {len(self.samples)} samples in Flickr30k {self.subset}')

        self.texts = []
        self.txt2img = {}
        self.img2txt = {}
        txt_id = 0

        for img_id, ann in enumerate(self.samples):
            self.img2txt[img_id] = []
            for caption in ann['caption']:
                self.texts.append(self.pre_caption(caption) + ' <SEP> ')
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1 
                
    def __len__(self):
        return len(self.samples)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n') 
        caption = caption.strip(' ')

        return caption

    def read_image(self, img_name):
        img = Image.open(os.path.join(self.info.img_dir, img_name)).convert('RGB')
        return img

    def __getitem__(self, i):
        sample = self.samples[i]
        
        img = self.read_image(os.path.join('..', sample['image']))
        query = self.texts
        answer = self.img2txt[i]

        img, _ = self.transform(img, None)

        #image, list of all texts, answer(caption id)
        return img, query, answer

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)

@hydra.main(config_path='../configs/task', config_name='vl_retrieval.yaml')
def main(cfg):
    dataset = Flickr30kDataset(cfg.dataset.info, 'val')
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    for data in dataloader:
        imgs, queries, answer_id = data
        print({
            'image': imgs,
            'text': queries,
            'answer_id': answer_id
        })
        break

if __name__ == '__main__':
    main()
