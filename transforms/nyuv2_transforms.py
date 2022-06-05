from .base import *


def make_nyuv2_transforms(image_set):
    normalize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    scales = [(224, 224)]
    resize_before_crop = [400, 500, 600]
    crop_size = 384
    # high resolution
    # scales = [(640, 640)]
    # resize_before_crop = [600, 700, 800]
    # crop_size = 512
    
    if image_set == "train":
        horizontal = [RandomHorizontalFlip()]
        return Compose(
            horizontal
            + [
                Compose([
                    RandomResize(resize_before_crop),
                    RandomSizeCrop(crop_size, max_size=resize_before_crop[-1], respect_boxes=False),
                    RandomResize(scales),
                ]),
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
