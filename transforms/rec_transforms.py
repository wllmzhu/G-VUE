from .base import *


def make_rec_transforms(phase):
    normalize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    scales = [(224, 224)]
    if phase == 'train':
        return Compose(
            [
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                RandomResize(scales),
                normalize,
            ]
        )
    else:
        return Compose(
            [
                RandomResize(scales),
                normalize,
            ]
        )
