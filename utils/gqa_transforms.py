from .transforms import *


def make_gqa_transforms(image_set, cautious):
    normalize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    scales = [(224, 224)]
    return Compose(
        [
            RandomResize(scales),
            normalize,
        ]
    )
