import random

import cv2
import numpy as np
import torch
from core import field
from FDA.utils import FDA_source_to_target_np
from skimage.io import imread
from torch.utils.data import Dataset


class PreCachedXview2Building(Dataset):
    def __init__(self, dataset, transforms=None):
        # self.target_fps = sorted(
        #     [fp for fp in glob.glob(os.path.join(targets_dir, "*.png")) if "pre" in fp]
        # )

        # self.image_fps = [
        #     os.path.join(image_dir, os.path.basename(fp.replace("_target.png", ".png")))
        #     for fp in self.target_fps
        # ]
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        img, y = self.dataset[idx]
        mask = y[field.MASK1]

        if self.transforms:
            blob = self.transforms(**dict(image=img, mask=mask))
            img = blob["image"]
            mask = blob["mask"]

        # y[field.MASK1] = mask
        # y["image_filename"] = os.path.basename(self.image_fps[idx])

        return self.split_image(img, y)

    def __len__(self):
        return len(self.dataset)

    def split_image(self, x, y):
        img = x
        mask = y[field.MASK1]
        c, h, w = img.shape
        mid_h, mid_w = h // 2, w // 2

        # Corners: a=top-left, b=top-right, c=bottom-left, d=bottom-right
        corner_a = img[:, :mid_h, :mid_w]
        corner_b = img[:, :mid_h, mid_w:]
        corner_c = img[:, mid_h:, :mid_w]
        corner_d = img[:, mid_h:, mid_w:]

        # Masks for each corner
        mask_a = mask[:mid_h, :mid_w]
        mask_b = mask[:mid_h, mid_w:]
        mask_c = mask[mid_h:, :mid_w]
        mask_d = mask[mid_h:, mid_w:]

        # Print shapes after splitting
        # print(f"Corner a shape: {corner_a.shape}")
        # print(f"Corner b shape: {corner_b.shape}")
        # print(f"Corner c shape: {corner_c.shape}")
        # print(f"Corner d shape: {corner_d.shape}")

        # print(f"Mask a shape: {mask_a.shape}")
        # print(f"Mask b shape: {mask_b.shape}")
        # print(f"Mask c shape: {mask_c.shape}")
        # print(f"Mask d shape: {mask_d.shape}")

        corners = {"a": corner_a, "b": corner_b, "c": corner_c, "d": corner_d}
        corner_masks = {"a": mask_a, "b": mask_b, "c": mask_c, "d": mask_d}

        # Randomly select base corner
        base_corner_key = random.choice(list(corners.keys()))
        base_corner = corners.pop(base_corner_key)
        base_mask = corner_masks[base_corner_key]

        # Randomly select helper corner
        helper_corner_key = random.choice(list(corners.keys()))
        helper_corner = corners[helper_corner_key]
        helper_mask = corner_masks[helper_corner_key]

        # (
        #     base_corner,
        #     base_mask,
        #     helper_corner,
        #     helper_mask,
        # ) = self.semantic_label_copy_paste_pair(
        #     base_corner, base_mask, helper_corner, helper_mask
        # )

        # Inside the split_image function
        base_corner = base_corner / 255
        helper_corner = helper_corner / 255
        base_mask = base_mask / 255
        helper_mask = helper_mask / 255

        x = torch.cat([base_corner, helper_corner], dim=0)
        y[field.MASK1] = base_mask
        y[field.VMASK2] = helper_mask

        return x, y

    ##### Strategy 3: copy & paste + Fourier blending
    def semantic_label_copy_paste_pair(
        self, base_corner, base_mask, helper_corner, helper_mask
    ):
        org_corner, org_mask = base_corner.copy(), base_mask.copy()
        # Get the indices of non-background pixels in the helper mask
        helper_indices = np.argwhere(helper_mask != 0)

        # Iterate over each non-background pixel in the helper mask
        for idx in helper_indices:
            x, y = idx

            # If the pixel is within the bounds of the base mask, copy the corresponding pixel from the helper corner
            if 0 <= x < base_mask.shape[0] and 0 <= y < base_mask.shape[1]:
                base_corner[x, y, :] = helper_corner[x, y, :]
                base_mask[x, y] = helper_mask[x, y]

        # Apply Fourier transform based blending
        blended_corner = self.fourier_blending(base_corner, helper_corner)

        # Adjust the helper mask accordingly
        blended_mask = base_mask.copy()

        return org_corner, org_mask, blended_corner, blended_mask

    def fourier_blending(self, source_image, target_image, L=0.01):
        # Ensure source and target images are in the same format
        source_image = source_image.transpose((2, 0, 1))
        target_image = target_image.transpose((2, 0, 1))

        # Apply FDA_source_to_target_np
        src_in_tar = FDA_source_to_target_np(source_image, target_image, L=L)

        # Transpose back to (H, W, C) format
        src_in_tar = src_in_tar.transpose((1, 2, 0))

        return src_in_tar
