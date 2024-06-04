import glob
import os
import random

import cv2
import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from core import field
from FDA.utils import FDA_source_to_target_np


class PreCachedXview2Building(Dataset):
    def __init__(self, image_dir, targets_dir, transforms=None, strategies=None):
        self.target_fps = sorted(
            [fp for fp in glob.glob(os.path.join(targets_dir, "*.png")) if "pre" in fp]
        )

        self.image_fps = [
            os.path.join(image_dir, os.path.basename(fp.replace("_target.png", ".png")))
            for fp in self.target_fps
        ]
        self.transforms = transforms
        self.strategies = strategies
        print(len(self.image_fps))

    def __getitem__(self, idx):
        x = imread(self.image_fps[idx])
        mask = imread(self.target_fps[idx])
        y = dict()

        # if self.transforms:
        #     blob = self.transforms(**dict(image=img, mask=mask))
        #     img = blob["image"]
        #     mask = blob["mask"]

        y[field.MASK1] = mask
        y["image_filename"] = os.path.basename(self.image_fps[idx])

        return self.split_image(x, y)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.image_fps)

    def split_image(self, x, y):
        """Splits image into 4 corners a,b,c,d

        Dimensions of each corner

        Image: Type: <class 'numpy.ndarray'> Shape: (512, 512, 3)

        Mask: Type: <class 'numpy.ndarray'> Shape: (512, 512)
        """
        img = x
        mask = y[field.MASK1]
        h, w, _ = img.shape
        mid_h, mid_w = h // 2, w // 2

        # Corners: a=top-left, b=top-right, c=bottom-left, d=bottom-right
        corner_a = img[:mid_h, :mid_w, :]
        corner_b = img[:mid_h, mid_w:, :]
        corner_c = img[mid_h:, :mid_w, :]
        corner_d = img[mid_h:, mid_w:, :]

        # Masks for each corner
        mask_a = mask[:mid_h, :mid_w]
        mask_b = mask[:mid_h, mid_w:]
        mask_c = mask[mid_h:, :mid_w]
        mask_d = mask[mid_h:, mid_w:]

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

        # Select random strategy to apply from activated strategies
        base_corner, base_mask, helper_corner, helper_mask = self.apply_random_strategy(
            base_corner, base_mask, helper_corner, helper_mask
        )

        # combine the images together.
        x = np.concatenate([base_corner, helper_corner], axis=2)

        # x = torch.cat([base_corner, helper_corner], dim=2).permute(2, 0, 1)
        y[field.MASK1] = base_mask
        y[field.VMASK2] = helper_mask

        return x, y

    def apply_random_strategy(self, base_corner, base_mask, helper_corner, helper_mask):
        """Applies a random strategy based on the enabled strategies"""
        # Define available strategies
        available_strategies = {
            "random_crop": self.random_crop,
            "semantic_label_inpainting_pair": self.semantic_label_inpainting_pair,
            "semantic_label_copy_paste_pair": self.semantic_label_copy_paste_pair,
        }

        # Filter strategies to only those that are enabled
        enabled_strategies = [key for key, value in self.strategies.items() if value]

        # Randomly select an enabled strategy
        selected_strategy = random.choice(enabled_strategies)
        strategy_method = available_strategies[selected_strategy]

        # Apply the selected strategy
        return strategy_method(base_corner, base_mask, helper_corner, helper_mask)

    ##### Strategy 1: Random Crop
    def random_crop(self, base_corner, base_mask, helper_corner, helper_mask):
        return base_corner, base_mask, helper_corner, helper_mask

    ##### Strategy 2: Inpainting
    def semantic_label_inpainting_pair(
        self, base_corner, base_mask, helper_corner, helper_mask
    ):
        """

        TODO  Possibly add a selection process in the corner selection for specific strategies so that it ensures the base_corner is not an empty image?
        """
        mask = base_mask.copy()
        # Find connected components in the mask
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

        # Calculate total image area
        total_area = mask.size

        # Remove the background component (label 0)
        object_stats = stats[1:]

        # Calculate areas of all objects
        object_areas = object_stats[:, 4]

        # Calculate the threshold area (30% of total image area)
        threshold_area = 0.3 * total_area

        # Exclude objects larger than the threshold area
        eligible_objects = [
            index for index, area in enumerate(object_areas) if area <= threshold_area
        ]
        print("mask dtype", mask.dtype)
        print("mask type: ", type(mask))
        print("Object areas: ", object_areas)
        print("Eligible objects:", eligible_objects)
        print("Number of eligible objects:", len(eligible_objects))

        if not eligible_objects:
            eligible_objects = list(range(len(object_areas)))

        # Randomly select objects to inpaint from eligible objects
        num_objects_to_remove = np.random.randint(1, len(eligible_objects) + 1)
        selected_object_indices = np.random.choice(
            eligible_objects, num_objects_to_remove, replace=False
        )

        # Create a binary mask for inpainting
        inpaint_mask = np.zeros_like(mask)
        for index in selected_object_indices:
            inpaint_mask[labels == index + 1] = 1

        print("inpainted_mask dtype", inpaint_mask.dtype)

        # Inpainting using OpenCV's inpaint method from TELEA
        inpainted_corner = cv2.inpaint(
            base_corner,
            inpaint_mask.astype(np.uint8),
            inpaintRadius=25,
            flags=cv2.INPAINT_TELEA,
        )

        # Update the mask by removing the selected objects
        updated_mask = np.where(inpaint_mask == 1, 0, mask)

        return base_corner, base_mask, inpainted_corner, updated_mask

    ##### Strategy 3: copy & paste + Fourier blending
    def semantic_label_copy_paste_pair(
        self, base_corner, base_mask, helper_corner, helper_mask
    ):
        """
        Performs the copy and paste strategy with Fourier blending.

        This method first performs a copy and paste of the two images,
        then uses Fourier blending to smooth out the edges and blend the image.

        Args:
            base_corner (np.ndarray): Base image corner.
            base_mask (np.ndarray): Mask for the base image.
            helper_corner (np.ndarray): Helper image corner.
            helper_mask (np.ndarray): Mask for the helper image.

        Returns:
            tuple: Original corner and mask, blended corner and mask.
        """
        # Copy the original base corner and mask
        org_corner, org_mask = base_corner.copy(), base_mask.copy()

        # Get the indices of non-background pixels in the helper mask
        helper_indices = np.argwhere(helper_mask != 0)

        # Iterate over each non-background pixel in the helper mask
        for idx in helper_indices:
            x, y = idx

            # Copy pixel from helper to base if within bounds
            if 0 <= x < base_mask.shape[0] and 0 <= y < base_mask.shape[1]:
                base_corner[x, y, :] = helper_corner[x, y, :]
                base_mask[x, y] = helper_mask[x, y]

        # Apply Fourier transform based blending
        blended_corner = self.fourier_blending(base_corner, helper_corner)

        # Adjust the helper mask accordingly
        blended_mask = base_mask.copy()

        # TODO Test if conversion of float64 to uint8 using this method improves performance
        # image = cv2.normalize(src=base_corner_np, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return org_corner, org_mask, blended_corner, blended_mask

    def fourier_blending(self, source_image, target_image, L=0.01):
        """
        Applies Fourier Domain Adaptation (FDA) to blend the source image with the target image.

        This method uses Fourier domain adaptation to transfer the spectral characteristics of the
        target image to the source image. The blending is controlled by the parameter L, which
        determines the extent of low-frequency component replacement.

        Args:
            source_image (numpy.ndarray): The source image, expected in (H, W, C) format.
            target_image (numpy.ndarray): The target image, expected in (H, W, C) format.
            L (float, optional): The low-frequency component scaling factor. Default is 0.01.

        Returns:
            numpy.ndarray: The blended image in (H, W, C) format.
        """
        # Ensure source and target images are in the same format
        source_image = source_image.transpose((2, 0, 1))
        target_image = target_image.transpose((2, 0, 1))

        # Apply FDA_source_to_target_np
        src_in_tar = FDA_source_to_target_np(source_image, target_image, L=L)

        # Transpose back to (H, W, C) format
        src_in_tar = src_in_tar.transpose((1, 2, 0))

        return src_in_tar
