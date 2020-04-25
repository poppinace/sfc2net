"""
@author: hao
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd 
import numpy as np
from PIL import Image
import cv2
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter
import h5py

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F


def read_image(x):
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:  # grayscale
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        h, w = image.shape[:2]
        crop_size = min(self.output_size, h, w)
        assert crop_size == self.output_size
        new_h = new_w = crop_size
        top = np.random.randint(0, h-new_h+1)
        left = np.random.randint(0, w-new_w+1)
        image = image[top:top+new_h, left:left+new_w, :]
        target = target[top:top+new_h, left:left+new_w]

        return {'image': image, 'target': target, 'gtcount': gtcount}


class RandomFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            target = cv2.flip(target, 1)
        return {'image': image, 'target': target, 'gtcount': gtcount}


class Normalize(object):

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        image, target = image.astype('float32'), target.astype('float32')

        # pixel normalization
        image = (self.scale * image - self.mean) / self.std

        image, target = image.astype('float32'), target.astype('float32')

        return {'image': image, 'target': target, 'gtcount': gtcount}


class ZeroPadding(object):
    def __init__(self, psize=32):
        self.psize = psize

    def __call__(self, sample):
        psize =  self.psize

        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        h,w = image.size()[-2:]
        ph,pw = (psize-h%psize),(psize-w%psize)
        # print(ph,pw)

        (pl, pr) = (pw//2, pw-pw//2) if pw != psize else (0, 0)
        (pt, pb) = (ph//2, ph-ph//2) if ph != psize else (0, 0)
        if (ph!=psize) or (pw!=psize):
            tmp_pad = [pl, pr, pt, pb]
            # print(tmp_pad)
            image = F.pad(image,tmp_pad)
            target = F.pad(target,tmp_pad)

        return {'image': image, 'target': target, 'gtcount': gtcount}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        image, target, gtcount = sample['image'], sample['target'], sample['gtcount']
        image = image.transpose((2, 0, 1))
        target = np.expand_dims(target, axis=2)
        target = target.transpose((2, 0, 1))
        image, target = torch.from_numpy(image), torch.from_numpy(target)
        return {'image': image, 'target': target, 'gtcount': gtcount}

class RiceDataset(Dataset):
    def __init__(self, data_dir, data_list, ratio, train=True, transform=None,gauss_kernel=8):
        self.gauss_kernel=gauss_kernel
        self.data_dir = data_dir
        self.data_list = [name.split('\t') for name in open(data_list).read().splitlines()]
        self.ratio = ratio
        self.train = train
        self.transform = transform
        
        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.gtcounts = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        if file_name[0] not in self.images:
            image = read_image(self.data_dir+file_name[0])
#            annotation=h5py.File(self.data_dir+file_name[1])
            annotation = sio.loadmat(self.data_dir+file_name[1])
            annotation= annotation['annPoints'][:]
#            annotation=np.transpose(annotation)
            h, w = image.shape[:2]
            r = 288. / min(h, w) if min(h, w) < 288 else self.ratio
            nh = int(np.ceil(h * r))
            nw = int(np.ceil(w * r))
            image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
            target = np.zeros((nh, nw), dtype=np.float32)
            if annotation is not None:
                pts = annotation
                gtcount = pts.shape[0]
                for pt in pts:
                    x, y = int(np.floor(pt[0] * r*1.0)), int(np.floor(pt[1] * r*1.0))
                    x, y = x - 1, y - 1
                    if x >= w or y >= h:
                        continue
                    target[y, x] = 1
            else:
                gtcount = 0
            target = gaussian_filter(target, self.gauss_kernel)

            # plt.imshow(target, cmap=cm.jet)
            # plt.show()
            # print(target.sum())

            self.images.update({file_name[0]:image})
            self.targets.update({file_name[0]:target})
            self.gtcounts.update({file_name[0]:gtcount})

        
        sample = {
            'image': self.images[file_name[0]], 
            'target': self.targets[file_name[0]], 
            'gtcount': self.gtcounts[file_name[0]]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample