from .base import *


def make_coco_transforms(image_set, cautious):
    normalize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    scales = [(224, 224)]
    resize_before_crop = [400, 500, 600]
    crop_size = 384
    
    if image_set == "train":
        horizontal = [RandomHorizontalFlip()]
        return Compose(
            horizontal
            + [
                Compose([
                    RandomResize(resize_before_crop),
                    RandomSizeCrop(crop_size, max_size=resize_before_crop[-1], respect_boxes=cautious),
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
