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
        each sample in self.samples consists of an image-text pair, plus an image_id
        return [
            [positive image, positive caption] x 1,
            [negative image, positive caption] x 7,
            [positive image, negative caption] x 7
        ]
        """
        sample = self.samples[i]
        img_id = sample['image_id']
        l = len(self.samples)

        # construct image pool
        select_img_ids = [img_id]
        neg_img_idxs = []
        while len(neg_img_idxs) < 7:
            neg_img_idx = randint(0, l-1)
            if self.samples[neg_img_idx]['image_id'] not in select_img_ids:
                neg_img_idxs.append(neg_img_idx)
        # extract images
        img = self.read_image(sample['image'])
        img, _ = self.transform(img, None)
        select_imgs = [img]
        for j in neg_img_idxs:
            select_imgs.append(
                self.transform(self.read_image(self.samples[j]['image']), None)[0]
            )
        # 8 -> 15
        select_imgs.extend([img]*7)

        # construct text pool
        select_img_ids = [img_id]
        neg_txt_idxs = []
        while len(neg_txt_idxs) < 7:
            neg_txt_idx = randint(0, l-1)
            if self.samples[neg_txt_idx]['image_id'] not in select_img_ids:
                neg_txt_idxs.append(neg_txt_idx)
        # extract captions
        caption = sample['caption']
        select_txts = [caption] * 8
        # 8 -> 15
        for j in neg_txt_idxs:
            select_txts.append(
                self.texts[j]
            )
        
        # return tensor [15, 3, H, W] for image
        return torch.stack(select_imgs), select_txts, 0
    
    def get_eval_samples(self, i):
        # return one image, all captions, and ground truth index
        sample = self.samples[i]
        img = self.read_image(sample['image'])
        img, _ = self.transform(img, None)

        # no caption returned, use self.img2txt to index self.texts instead
        return img, None, i

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
