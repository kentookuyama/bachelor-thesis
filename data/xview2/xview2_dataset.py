import glob
import os
import random

import cv2
import numpy as np
import torch
from albumentations import Compose, HorizontalFlip, RandomCrop, RandomRotate90
from skimage.io import imread
from torch.utils.data import Dataset

from core import field
from FDA.utils import FDA_source_to_target_np


class PreCachedXview2Building(Dataset):
    def __init__(self, image_dir, targets_dir, transforms=None, strategies=None):
        if image_dir == "./xview2/tier3/images":
            self.image_fps = sorted(
                [
                    fp
                    for fp in glob.glob(os.path.join(image_dir, "*.png"))
                    if "pre" in fp
                ]
            )

            base_image_filenames = set(
                os.path.basename(fp).replace("_pre_disaster.png", "")
                for fp in self.image_fps
            )

            constructed_target_fps = sorted(
                os.path.join(
                    targets_dir,
                    f"localization_{base_name.replace('_', '-')}_target.png",
                )
                for base_name in base_image_filenames
            )

            self.target_fps = [
                fp for fp in constructed_target_fps if os.path.isfile(fp)
            ]
        else:
            self.target_fps = sorted(
                [
                    fp
                    for fp in glob.glob(os.path.join(targets_dir, "*.png"))
                    if "pre" in fp
                ]
            )

            self.image_fps = [
                os.path.join(
                    image_dir, os.path.basename(fp.replace("_target.png", ".png"))
                )
                for fp in self.target_fps
            ]
        self.transforms = transforms
        self.strategies = strategies
        self.skipped_amount = 0
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
        if self.strategies is None:
            return x, y

        # img uint8 mask uint8

        # Apply inital geo_transforms here
        if self.transforms:
            augmented = self.transforms(**dict(image=x, mask=mask))
            y[field.MASK1] = augmented["mask"]
            x = augmented["image"]

        # img uint8 mask uint8

        return self.split_image(x, y)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.image_fps)

    def split_image(self, x, y):
        """Splits image into corners and applies random cropping and flipping based on image size.

        Incoming image size variations:
            - image shape(1280, 1280, 3) dtype uint8
            - mask shape(1280, 1280) dtype uint8
            - image shape(1536, 1536, 3) dtype uint8
            - mask shape(1536, 1536) dtype uint8
            - image shape(768, 768, 3) dtype uint8
            - mask shape(768, 768) dtype uint8

        Returns:
            Cropped and possibly flipped image and mask pairs.
        """

        ## First select strategy
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

        img = x
        mask = y[field.MASK1]
        h, w, _ = img.shape

        if (h, w) == (768, 768):
            # Handle 768x768 images by first splitting into 4 corners of 512x512
            mid_h, mid_w = 512, 512

            base_corner, base_mask, helper_corner, helper_mask = self.corner_selection(
                img, mask, mid_w, mid_h, selected_strategy
            )

            # shape: (512, 512, 3) dtype : uint8

            # Apply random flipping and rotating to reduce similarities
            helper_corner, helper_mask = self.apply_transformations(
                helper_corner,
                helper_mask,
                [HorizontalFlip(p=0.5), RandomRotate90(p=0.5)],
            )

            # shape: (512, 512, 3) dtype : uint8

        else:
            # Handle other image sizes by splitting into 4 quarters
            mid_h, mid_w = h // 2, w // 2

            base_corner, base_mask, helper_corner, helper_mask = self.corner_selection(
                img, mask, mid_w, mid_h, selected_strategy
            )

            base_corner, base_mask = self.apply_transformations(
                base_corner, base_mask, [RandomCrop(512, 512, always_apply=True)]
            )
            helper_corner, helper_mask = self.apply_transformations(
                helper_corner, helper_mask, [RandomCrop(512, 512, always_apply=True)]
            )

        # Apply selected strategy method
        base_corner, base_mask, helper_corner, helper_mask = strategy_method(
            base_corner, base_mask, helper_corner, helper_mask
        )

        # dtype uint8 | shape (512, 512, 3) | (512, 512)

        y[field.MASK1] = base_mask
        y[field.VMASK2] = helper_mask

        x = np.concatenate([base_corner, helper_corner], axis=2)

        return x, y

    def apply_transformations(self, image, mask, transformations):
        """
        applies transformations to the image and mask

        image, mask dtype : uint8
        """

        composed_transform = Compose(transformations)
        augmented = composed_transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]

    def corner_selection(self, img, mask, mid_w, mid_h, strategy):
        # Corners: a=top-left, b=top-right, c=bottom-left, d=bottom-right
        corner_a = img[:mid_h, :mid_w, :]
        corner_b = img[:mid_h, -mid_w:, :]
        corner_c = img[-mid_h:, :mid_w, :]
        corner_d = img[-mid_h:, -mid_w:, :]

        # Masks for each corner
        mask_a = mask[:mid_h, :mid_w]
        mask_b = mask[:mid_h, -mid_w:]
        mask_c = mask[-mid_h:, :mid_w]
        mask_d = mask[-mid_h:, -mid_w:]

        corners = {
            "a": (corner_a, mask_a),
            "b": (corner_b, mask_b),
            "c": (corner_c, mask_c),
            "d": (corner_d, mask_d),
        }

        non_empty_corners = [
            key for key, (corner, mask) in corners.items() if np.sum(mask) > 0
        ]

        if strategy == "semantic_label_inpainting_pair":
            base_corner_key = (
                random.choice(non_empty_corners)
                if non_empty_corners
                else random.choice(list(corners.keys()))
            )
            remaining_corners = [
                key for key in corners.keys() if key != base_corner_key
            ]
            helper_corner_key = random.choice(remaining_corners)

        # Additional logic based on selected strategy
        if strategy == "semantic_label_inpainting_pair":
            base_corner_key = (
                random.choice(non_empty_corners)
                if non_empty_corners
                else random.choice(list(corners.keys()))
            )
            remaining_corners = [
                key for key in corners.keys() if key != base_corner_key
            ]
            helper_corner_key = random.choice(remaining_corners)

        elif strategy == "semantic_label_copy_paste_pair":
            helper_corner_key = (
                random.choice(non_empty_corners)
                if non_empty_corners
                else random.choice(list(corners.keys()))
            )
            remaining_corners = [
                key for key in corners.keys() if key != helper_corner_key
            ]
            base_corner_key = random.choice(remaining_corners)

        else:
            base_corner_key = random.choice(list(corners.keys()))
            remaining_corners = [
                key for key in corners.keys() if key != base_corner_key
            ]
            helper_corner_key = random.choice(remaining_corners)

        # Obtain base corner and helper corner based on the selected keys
        base_corner, base_mask = corners[base_corner_key]
        helper_corner, helper_mask = corners[helper_corner_key]

        return base_corner, base_mask, helper_corner, helper_mask

    ##### Preliminary Check of eligibility to strategy.
    def eligible_image(self, mask):
        mask = mask.copy()
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
        return eligible_objects, labels

    ##### Strategy 1: Random Crop
    def random_crop(self, base_corner, base_mask, helper_corner, helper_mask):
        return self.random_rotation(base_corner, base_mask, helper_corner, helper_mask)

    def random_rotation(self, base_corner, base_mask, helper_corner, helper_mask):
        random_rotate = RandomRotate90(True)
        rotated_image = random_rotate(image=helper_corner, mask=helper_mask)
        return base_corner, base_mask, rotated_image["image"], rotated_image["mask"]

    ##### Strategy 2: Inpainting
    def semantic_label_inpainting_pair(
        self, base_corner, base_mask, helper_corner, helper_mask
    ):
        """
        Perform semantic label inpainting on the base corner using objects from the helper corner.

        - Identify eligible objects from the helper corner that can be used for inpainting.
        - If no eligible objects are found, perform a random crop instead.
        - Randomly select a subset of eligible objects for inpainting.
        - Create a binary mask for the selected objects.
        - Inpaint the selected objects into the base corner using OpenCV's inpaint method.
        - Update the mask to reflect the inpainting.

        Args:
            base_corner (np.ndarray): The base image corner to be inpainted.
            base_mask (np.ndarray): The mask corresponding to the base image corner.
            helper_corner (np.ndarray): The helper image corner providing objects for inpainting.
            helper_mask (np.ndarray): The mask corresponding to the helper image corner.

        Returns:
            tuple: The original base corner and mask, the inpainted corner, and the updated mask.
        """
        eligible_objects, labels = self.eligible_image(base_mask)

        # If object sizes are larger than 30% in image size and no objects are eligible, use random_crop
        if not eligible_objects:
            self.skipped_amount += 1
            if self.skipped_amount % 200 == 0:
                print(f"Skipped {self.skipped_amount} times.")
            return self.random_crop(base_corner, base_mask, helper_corner, helper_mask)

        # Randomly select objects to inpaint from eligible objects
        num_objects_to_remove = np.random.randint(1, len(eligible_objects) + 1)
        selected_object_indices = np.random.choice(
            eligible_objects, num_objects_to_remove, replace=False
        )

        # Create a binary mask for inpainting
        inpaint_mask = np.zeros_like(base_mask)
        for index in selected_object_indices:
            inpaint_mask[labels == index + 1] = 1

        # Inpainting using OpenCV's inpaint method from TELEA
        inpainted_corner = cv2.inpaint(
            base_corner,
            inpaint_mask,
            inpaintRadius=25,
            flags=cv2.INPAINT_TELEA,
        )

        # Update the mask by removing the selected objects
        updated_mask = np.where(inpaint_mask == 1, 0, base_mask)

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

        # Get eligible objects from the helper_corner
        helper_eligible_objects, helper_labels = self.eligible_image(helper_mask)

        # If object sizes are larger than 30% in image size and no objects are eligible, use random_crop
        if not helper_eligible_objects:
            self.skipped_amount += 1
            if self.skipped_amount % 200 == 0:
                print(f"Skipped {self.skipped_amount} times.")
            return self.random_crop(base_corner, base_mask, helper_corner, helper_mask)

        # Create Binary mask for inpainting eligible objects from the helper mask
        inpaint_mask = np.zeros_like(base_mask)
        for obj_index in helper_eligible_objects:
            inpaint_mask[helper_labels == obj_index + 1] = 1

        # Get the indices of non-background pixels in the helper mask
        helper_indices = np.argwhere(inpaint_mask != 0)

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

        # Convert Float64 to Uint8
        src_in_tar = cv2.normalize(
            src=src_in_tar,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        return src_in_tar
