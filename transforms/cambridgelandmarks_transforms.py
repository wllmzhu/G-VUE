from .base import *
import torchvision.transforms as T

class CustomCenterCrop(object):
    def __init__(self, size):
        self.centercrop = T.CenterCrop(size)

    def __call__(self, img, target):
        return self.centercrop(img), None if target is None else self.centercrop(target)

def make_cambridgelandmarks_transforms():
    normalize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    scales = [(224, 224)]
    crop_size = [1080, 1080]

    return Compose(
        [
            CustomCenterCrop(crop_size),
            RandomResize(scales),
            normalize,
        ]
    )
    