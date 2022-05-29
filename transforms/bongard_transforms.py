from .base import *


def make_bongard_transforms(image_set):
    normalize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    scales = [(224, 224)]
    # ColorJitter parameters from Bongard-HOI
    colorjitters = {'brightness':0.4, 'contrast':0.4, 'saturation':0.4, 'hue':0.5/3.14}
    
    if image_set == "train":
        horizontal = [RandomHorizontalFlip()]
        return Compose(
            horizontal
            + [
                RandomResize([scales[-1]]),
                ColorJitter(colorjitters),
                normalize,
            ]
        )
    else:
        return Compose(
            [
                RandomResize([scales[-1]]),
                normalize,
            ]
        )
    