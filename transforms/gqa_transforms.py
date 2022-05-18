from .base import *


def make_gqa_transforms():
    normalize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return Compose(
        [
            normalize,
        ]
    )
