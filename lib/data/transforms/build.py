# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing, RandomPatch, ColorSpaceConvert, ColorAugmentation, RandomBlur, GaussianBlur
from .augmix import AugMix
from .autoaug import ImageNetPolicy

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    # Here we must notice that adam optimizer was sensitive with the large color jitter and random affinity operations

    if is_train:
        #transform = T.Compose( [ T.Resize(cfg.INPUT.SIZE_TRAIN), T.ToTensor() ])

        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            RandomPatch(prob_happen=cfg.INPUT.RANDOM_PATCH_PROB, patch_max_area=0.16),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)], p=cfg.INPUT.COLORJIT_PROB),
            #ImageNetPolicy(prob=cfg.INPUT.AUTOAUG_PROB),
            #T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0)], p=cfg.INPUT.COLORJIT_PROB),

            T.RandomApply([T.transforms.RandomAffine(degrees=0, scale=(0.9, 1.1),shear=None, resample=False, fillcolor=128)], p=cfg.INPUT.RANDOM_AFFINE_PROB),
            #T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
            #GaussianBlur(p=cfg.INPUT.RANDOM_BLUR_PROB),

            #AugMix(prob=cfg.INPUT.AUGMIX_PROB),
            #RandomBlur(p=cfg.INPUT.RANDOM_BLUR_PROB),

            # Here we found that the color jitter and random-affine can not boost the performance.
            #T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            #T.transforms.RandomAffine(30, translate=None, scale=[0.9, 1.1], shear=None, resample=False,fillcolor=128),
            T.ToTensor(),
            normalize_transform,
            #ColorAugmentation(),
            RandomErasing(probability=cfg.INPUT.RE_PROB, sh=cfg.INPUT.RE_SH, mean=cfg.INPUT.PIXEL_MEAN),

        ])

    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.ToTensor(),
            normalize_transform

        ])

    return transform
