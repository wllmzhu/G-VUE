import os
import hydra
import json
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset
import utils.io as io
from utils.misc import collate_fn
from base import DATASET
from transforms.vcr_transforms import color_list, OPACITY
#from transforms.vcr_transforms import make_vcr_transforms


@DATASET.register()
class VCRDataset(Dataset):
    def __init__(self, info, subset):
        super().__init__()
        self.info = info
        self.subsets = ['train', 'val', 'test']
        self.subset = subset
        assert self.subset in self.subsets, f'subset {self.subset} not in {self.subsets} (test is not a valid split for GQA because it contains questions only)'
        #self.transform = make_vcr_transforms()
        self._load_dataset()

    def _load_dataset(self):
        with open(os.path.join(self.info.anno_dir, f'{self.subset}.jsonl'), 'r') as f:
            self.samples = f.readlines()

        print(f'load {len(self.samples)} samples in VCR {self.subset}')

                
    def __len__(self):
        return len(self.samples)

    def read_image(self, img_name):
        img = Image.open(os.path.join(self.info.img_dir, img_name)).convert('RGBA')
        return img

    def item_to_str(self, mixed_list):
        return [str(item) for item in mixed_list]

    def overlay_bbox(self, image, bboxes, names):

        for i, box in enumerate(bboxes):
            if names[i] == 'person':
                color = color_list[:-1][i % (len(color_list) - 1)]
            else:
                color = color_list[-1]

            box = [int(x) for x in box[:4]]
            x1, y1, x2, y2 = box
            shape = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]

            overlay = Image.new('RGBA', image.size, tuple(color) + (0,))
            draw = ImageDraw.Draw(overlay)
            draw.polygon(shape, fill=tuple(color) + (OPACITY,))
            draw.line(shape, fill=tuple(color), width=7)

            image = Image.alpha_composite(image, overlay)

        image = image.convert("RGB")

        return image

    def __getitem__(self, i):
        sample = json.loads(self.samples[i])

        choices = [self.item_to_str(choice) for choice in sample["answer_choices"]]
        choices = ' [SEP] '.join([' '.join(choice) for choice in choices])
        question = ' '.join(self.item_to_str(sample['question']))
        question += ' [SEP] ' + choices

        answer_index = sample['answer_label']

        img = self.read_image(sample['img_fn'])

        metadata = io.load_json_object(os.path.join(self.info.img_dir, sample['metadata_fn']))
        bboxes = metadata['boxes']
        names = metadata['names']

        img = self.overlay_bbox(img, bboxes, names)

        #img, _ = self.transform(img, None)

        #image, question + 4 candidate choices, answer(choice id)
        return img, question, answer_index

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate_fn, **kwargs)

@hydra.main(config_path='../configs/task', config_name='common_sense.yaml')
def main(cfg):
    dataset = VCRDataset(cfg.dataset.info, 'val')
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    for data in dataloader:
        imgs, queries, answer_id = data
        print({
            'text': queries,
            'answer_id': answer_id
        })
        imgs[0].show()
        break

if __name__ == '__main__':
    main()
