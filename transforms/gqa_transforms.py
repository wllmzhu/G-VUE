# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for image, bbox and mask.
"""
from .base_transforms import Compose, ToTensor, Normalize


def make_gqa_transforms():
    normalize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return Compose(
        [
            normalize,
        ]
    )

