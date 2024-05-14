import glob
import os
import random

import numpy as np
import torch
from core import field
from skimage.io import imread
from torch.utils.data import Dataset


class ColorAugDataset(Dataset):
    def __init__(
        self, image_dir, targets_dir, geo_transform, color_transform, common_transform
    ):
        self.target_fps = sorted(
            [fp for fp in glob.glob(os.path.join(targets_dir, "*.png")) if "pre" in fp]
        )

        self.image_fps = [
            os.path.join(image_dir, os.path.basename(fp.replace("_target.png", ".png")))
            for fp in self.target_fps
        ]
        self.color_transform = color_transform
        self.geo_transform = geo_transform
        self.common_transform = common_transform

    def __getitem__(self, idx):
        x = imread(self.image_fps[idx])
        mask = imread(self.target_fps[idx])
        y = dict()
        y[field.MASK1] = mask
        y["image_filename"] = os.path.basename(self.image_fps[idx])

        # Get initial image and mask shapes for comparison
        orig_img_shape = x.shape
        orig_mask_shape = y[field.MASK1].shape

        blob = self.geo_transform(**dict(image=x, mask=y[field.MASK1]))
        img = blob["image"]
        mask = blob["mask"]

        # Check image and mask shapes after geometric transformation
        if img.shape != orig_img_shape or mask.shape != orig_mask_shape:
            print(f"WARNING: Shape mismatch after geo_transform!")
            print(f"  Original image: {orig_img_shape}, mask: {orig_mask_shape}")
            print(f"  After geo_transform: image: {img.shape}, mask: {mask.shape}")

        # x, mask -> tensor
        blob = self.common_transform(image=img, mask=mask)
        org_img = blob["image"]
        mask = blob["mask"]
        y[field.MASK1] = mask

        # x -> color_trans_x -> tensor
        if self.color_transform:
            color_trans_x = self.color_transform(**dict(image=img))["image"]
            blob = self.common_transform(image=color_trans_x)
            color_trans_x = blob["image"]
            y[field.COLOR_TRANS_X] = color_trans_x

        return org_img, y

    def __len__(self):
        return len(self.image_fps)
