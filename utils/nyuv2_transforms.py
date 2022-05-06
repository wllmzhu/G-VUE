import random
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from utils.misc import interpolate

def make_nyuv2_transforms(image_set):
    return Compose([])
    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
