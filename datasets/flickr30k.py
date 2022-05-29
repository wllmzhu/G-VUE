import os
from random import randint, choice
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

        for i, ann in enumerate(self.samples):
            if 'image_id' in ann:
                img_id = ann['image_id']
            else:
                img_id = i
            
            if img_id not in self.img2txt:
                self.img2txt[img_id] = []
            
            cands = ann['caption'] if isinstance(ann['caption'], list) else [ann['caption']]
            for caption in cands:
                self.texts.append(self.pre_caption(caption))
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
        img = Image.open(os.path.join(self.info.anno_dir, img_name)).convert('RGB')
        return img

    def __getitem__(self, i):
        if self.subset == 'train':
            return self.get_learn_samples(i)
        else:
            return self.get_eval_samples(i)
    
    def get_learn_samples(self, i):
        """
        return [
            [positive image, positive caption],
            [negative image, positive caption],
            [positive image, negative caption 1],
            [positive image, negative caption 2]
        ]
        """
        sample = self.samples[i]
        img = self.read_image(sample['image'])
        caption = sample['caption']
        img_id = sample['image_id']

        neg_img_i = i
        while self.samples[neg_img_i]['image_id'] == img_id:
            neg_img_i = randint(0, len(self.samples)-1)
        neg_img = self.read_image(self.samples[neg_img_i]['image'])

        neg_txt1_i = i
        while self.txt2img[neg_txt1_i] == img_id:
            neg_txt1_i = randint(0, len(self.texts)-1)
        neg_txt1 = self.texts[neg_txt1_i]

        neg_txt2_i = i
        while self.txt2img[neg_txt2_i] == img_id:
            neg_txt2_i = randint(0, len(self.texts)-1)
        neg_txt2 = self.texts[neg_txt2_i]

        img, _ = self.transform(img, None)
        neg_img, _ = self.transform(neg_img, None)
        
        # return tensor [4, 3, H, W] for image
        return torch.stack([img, neg_img, img, img]), [caption, caption, neg_txt1, neg_txt2], 0
    
    def get_eval_samples(self, i):
        # return one image, all captions, and ground truth index
        sample = self.samples[i]
        img = self.read_image(sample['image'])
        img, _ = self.transform(img, None)

        # sample 1 caption out of 5 candidates
        queries = []
        for j in range(len(self.img2txt)):
            cands = self.img2txt[j]
            queries.append(self.texts[choice(cands)])

        return img, queries, i

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
