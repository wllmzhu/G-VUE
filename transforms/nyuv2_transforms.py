import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import random

from .base_transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResize, RandomSizeCrop

def make_nyuv2_transforms(image_set, cautious):
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



        # if self.args.do_kb_crop is True:
        #     height = image.height
        #     width = image.width
        #     top_margin = int(height - 352)
        #     left_margin = int((width - 1216) / 2)
        #     depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
        #     image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

        # if self.args.do_random_rotate is True:
        #     random_angle = (random.random() - 0.5) * 2 * self.args.degree
        #     image = rotate_image(image, random_angle)
        #     depth_gt = rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
        
        # img, depth_gt = random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
        # image, depth_gt = train_preprocess(image, depth_gt)



# def rotate_image(self, image, angle, flag=Image.BILINEAR):
#         result = image.rotate(angle, resample=flag)
#         return result

# def random_crop(self, img, depth, height, width):
#     assert img.shape[0] >= height
#     assert img.shape[1] >= width
#     assert img.shape[0] == depth.shape[0]
#     assert img.shape[1] == depth.shape[1]
#     x = random.randint(0, img.shape[1] - width)
#     y = random.randint(0, img.shape[0] - height)
#     img = img[y:y + height, x:x + width, :]
#     depth = depth[y:y + height, x:x + width, :]
#     return img, depth

# def train_preprocess(self, image, depth_gt):
#     # Random flipping
#     do_flip = random.random()
#     if do_flip > 0.5:
#         image = (image[:, ::-1, :]).copy()
#         depth_gt = (depth_gt[:, ::-1, :]).copy()

#     # Random gamma, brightness, color augmentation
#     do_augment = random.random()
#     if do_augment > 0.5:
#         image = self.augment_image(image)

#     return image, depth_gt

# def augment_image(self, image):
#     # gamma augmentation
#     gamma = random.uniform(0.9, 1.1)
#     image_aug = image ** gamma

#     # brightness augmentation
#     if self.args.dataset == 'nyu':
#         brightness = random.uniform(0.75, 1.25)
#     else:
#         brightness = random.uniform(0.9, 1.1)
#     image_aug = image_aug * brightness

#     # color augmentation
#     colors = np.random.uniform(0.9, 1.1, size=3)
#     white = np.ones((image.shape[0], image.shape[1]))
#     color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
#     image_aug *= color_image
#     image_aug = np.clip(image_aug, 0, 1)

#     return image_aug



# class ToTensor(object):
#     def __init__(self, mode):
#         self.mode = mode
#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
#     def __call__(self, sample):
#         image = sample['image']
#         image = self.to_tensor(image)
#         image = self.normalize(image)

#         depth = sample['depth']
#         depth = self.to_tensor(depth)
#         return {'image': image, 'depth': depth}
    
#     def to_tensor(self, pic):
#         if isinstance(pic, np.ndarray):
#             img = torch.from_numpy(pic.transpose((2, 0, 1)))
#             return img
        
#         # handle PIL Image
#         if pic.mode == 'I':
#             img = torch.from_numpy(np.array(pic, np.int32, copy=False))
#         elif pic.mode == 'I;16':
#             img = torch.from_numpy(np.array(pic, np.int16, copy=False))
#         else:
#             img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
#         # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
#         if pic.mode == 'YCbCr':
#             nchannel = 3
#         elif pic.mode == 'I;16':
#             nchannel = 1
#         else:
#             nchannel = len(pic.mode)
#         img = img.view(pic.size[1], pic.size[0], nchannel)
        
#         img = img.transpose(0, 1).transpose(0, 2).contiguous()
#         if isinstance(img, torch.ByteTensor):
#             return img.float()
#         else:
#             return img
