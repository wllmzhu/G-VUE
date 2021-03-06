import numpy as np
from .base import *


def make_vcr_transforms():
    normalize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    scales = [(224, 224)]
    return Compose(
        [
            RandomResize(scales),
            normalize,
        ]
    )

# Overlay parameters
TRANSPARENCY = .15
OPACITY = int(255 * TRANSPARENCY)

color_list = np.array(
    [
        [255, 0, 0],
        [0, 255, 0],
        [236, 176, 31],
        [0, 0, 255],
        [255, 0, 255],
        [170, 0, 255],
        [255, 255, 0],
        [170, 84, 0],
        [84, 84, 0],
        [255, 127, 0],
        [76, 189, 237],
        [170, 0, 127],
        [125, 46, 141],
        [190, 190, 0],
        [161, 19, 46],
        [0, 170, 127],
        [255, 170, 127],
        [0, 84, 127],
        [255, 84, 127],
        [170, 170, 255],
        [170, 170, 127],
        [84, 0, 0],
        [0, 170, 0],
        [0, 255, 255],
        [255, 170, 255],
        [84, 0, 127],
        [255, 255, 127],
        [170, 0, 0],
        [84, 255, 127],
        [0, 0, 127],
        [170, 84, 127],
        [170, 84, 255],
        [170, 170, 0],
        [216, 82, 24],
        [0, 84, 0],
        [84, 0, 255],
        [255, 0, 127],
        [127, 0, 0],
        [170, 255, 127],
        [170, 255, 255],
        [0, 127, 0],
        [0, 0, 170],
        [84, 170, 127],
        [0, 113, 188],
        [118, 171, 47],
        [84, 84, 127],
        [0, 42, 0],
        [84, 84, 255],
        [84, 170, 0],
        [84, 170, 255],
        [170, 255, 0],
        [0, 0, 212],
        [0, 212, 0],
        [0, 0, 84],
        [0, 84, 255],
        [145, 145, 145]
    ]
)
