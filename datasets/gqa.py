import os
import hydra
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import utils.io as io
from utils.misc import collate_fn
from .base import DATASET
from transforms.gqa_transforms import make_gqa_transforms


@DATASET.register()
class GQADataset(Dataset):
    def __init__(self, info, subset):
        super().__init__()
        self.info = info
        self.subset = subset
        self.subsets = info.subsets
        assert self.subset in self.subsets, f'subset {self.subset} not in {self.subsets} (test is not a valid split for GQA because it contains questions only)'
        self.transform = make_gqa_transforms()
        self._load_dataset()
        self._build_dict()

    def _load_dataset(self):
        self.samples = io.load_json_object(
            os.path.join(self.info.anno_dir, f'{self.subset}_balanced_questions.json')
        )
        print(f'load {len(self.samples)} samples in GQA {self.subset}')

        # i-th entry to entry key, e.g. 0 -> '201307251', 1 -> '201640614'
        self.i_to_key = {}
        cur_i = 0
        for key, _ in self.samples.items():
            # Record entry key mapping 
            self.i_to_key[cur_i] = key
            cur_i += 1

    def _build_dict(self):
        # Answer to answer ID, e.g. 'yes' -> 0, 'cat' -> 1.
        self.answer_to_idx = {}
        self.idx_to_answer = {}
        cur_idx = 0
        for subset in self.subsets:
            if subset == self.subset:
                split = self.samples
            else:
                split = io.load_json_object(
                    os.path.join(self.info.anno_dir, f'{subset}_balanced_questions.json')
                )
                print(f'(building answer ID dictionary) load {len(split)} samples in {subset}')
            for _, sample in split.items():
                answer = sample['answer']
                if answer not in self.answer_to_idx:
                    # Record answer ID mapping
                    self.answer_to_idx[answer] = cur_idx
                    self.idx_to_answer[cur_idx] = answer
                    cur_idx += 1
                
    def __len__(self):
        return len(self.samples)

    def read_image(self, img_name):
        img = Image.open(os.path.join(self.info.img_dir, img_name)).convert('RGB')
        return img

    def __getitem__(self, i):
        sample = self.samples[self.i_to_key[i]]
        
        img = self.read_image('{}.jpg'.format(sample['imageId']))
        question = sample['question'].lower().replace(',', '').replace('.', '').replace('?', '').replace('\'s', ' \'s')
        answer = sample['answer']

        img, _ = self.transform(img, None)

        #image, question, answer(vocab id)
        return img, question, self.answer_to_idx[answer]

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)

@hydra.main(config_path='../configs/task', config_name='vqa.yaml')
def main(cfg):
    dataset = GQADataset('gqa', cfg.dataset.info, 'val')
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
