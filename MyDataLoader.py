import glob
import io
import numpy as np
import re
import os
from io import BytesIO
import random
from uuid import uuid4
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import to_tensor


class ListDatasetLite(Dataset):
    def __init__(self, root, list_file, patch_size=96, shrink_size=2, noise_level=1, down_sample_method=None, transform=None):
        self.root = root
        self.transform = transform
        self.random_cropper = RandomCrop(size=patch_size)
        self.img_augmenter = ImageAugment(shrink_size, noise_level, down_sample_method)
        self.transform = transform
        self.fnames = []

        if isinstance(list_file, list):
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            self.fnames.append(line)


    def __getitem__(self, idx):
        fname = self.fnames[idx].strip()
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_patch = self.random_cropper(img)
        lr_img, hr_img = self.img_augmenter.process(img_patch)
        return self.transform(lr_img), self.transform(hr_img)
        #return to_tensor(lr_img), to_tensor(hr_img)

    def __len__(self):
        return self.num_imgs



class ImageAugment:
    def __init__(self,
                 shrink_size=2,
                 noise_level=1,
                 down_sample_method=None
                 ):
        # noise_level (int): 0: no noise; 1: 75-95% quality; 2:50-75%
        if noise_level == 0:
            self.noise_level = [0, 0]
        elif noise_level == 1:
            self.noise_level = [5, 25]
        elif noise_level == 2:
            self.noise_level = [25, 50]
        else:
            raise KeyError("Noise level should be either 0, 1, 2")
        self.shrink_size = shrink_size
        self.down_sample_method = down_sample_method

    def shrink_img(self, hr_img):

        if self.down_sample_method is None:
            resample_method = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
        else:
            resample_method = self.down_sample_method
        img_w, img_h = tuple(map(lambda x: int(x / self.shrink_size), hr_img.size))
        lr_img = hr_img.resize((img_w, img_h), resample_method)
        return lr_img

    def add_jpeg_noise(self, hr_img):
        quality = 100 - round(random.uniform(*self.noise_level))
        lr_img = BytesIO()
        hr_img.save(lr_img, format='JPEG', quality=quality)
        lr_img.seek(0)
        lr_img = Image.open(lr_img)
        return lr_img

    def process(self, hr_patch_pil):
        lr_patch_pil = self.shrink_img(hr_patch_pil)
        if self.noise_level[1] > 0:
            lr_patch_pil = self.add_jpeg_noise(lr_patch_pil)

        return lr_patch_pil, hr_patch_pil

    def up_sample(self, img, resample):
        width, height = img.size
        return img.resize((self.shrink_size * width, self.shrink_size * height), resample=resample)
