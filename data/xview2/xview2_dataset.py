import glob
import os
import random

import cv2
import numpy as np
import torch

# from albumentations import Compose, HorizontalFlip, RandomCrop, RandomRotate90
from albumentations import *
from core import field
from FDA.utils import FDA_source_to_target_np
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset


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
        self.skipped_amount = 0
        print(len(self.image_fps))

    def __getitem__(self, idx):
        x = imread(self.image_fps[idx])
        mask = imread(self.target_fps[idx])
        y = dict()

        self.check_data(x, np.uint8, (1024, 1024, 3))
        self.check_data(mask, np.uint8, (1024, 1024))

        y[field.MASK1] = mask
        y["image_filename"] = os.path.basename(self.image_fps[idx])

        if self.strategies is None:
            return x, y

        if self.transforms:
            x, y[field.MASK1] = self.apply_transforms(x, y[field.MASK1])

        # img uint8 mask uint8

        return self.split_image(x, y)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.image_fps)

    ### Methods for creating the dataset

    def apply_transforms(self, image, mask):
        """Applies the transforms to the image and mask"""
        image = image.astype(np.float32)
        mask = mask

        augmented = self.transforms(**dict(image=image, mask=mask))
        mask = augmented["mask"]
        image = augmented["image"]

        # self.check_data(image, np.float32, (1024, 1024, 3))
        # self.check_data(mask, np.float32, (1024, 1024))

        # scale_factor = np.random.choice([0.75, 1.25, 1.5])
        # image = cv2.resize(
        #     image,
        #     None,
        #     fx=scale_factor,
        #     fy=scale_factor,
        #     interpolation=cv2.INTER_NEAREST,
        # )
        # mask = cv2.resize(
        #     mask,
        #     None,
        #     fx=scale_factor,
        #     fy=scale_factor,
        #     interpolation=cv2.INTER_NEAREST,
        # )

        # self.check_data(
        #     image, np.float32, (1024 * scale_factor, 1024 * scale_factor, 3)
        # )
        # self.check_data(mask, np.float32, (1024 * scale_factor, 1024 * scale_factor))

        return image, mask

    def split_image(self, x, y):
        """Splits image into corners and applies random cropping and flipping based on image size.

        Incoming image size variations:
            - image shape(1280, 1280, 3) dtype float32
            - mask shape(1280, 1280) dtype float32
            - image shape(1536, 1536, 3) dtype float32
            - mask shape(1536, 1536) dtype float32
            - image shape(768, 768, 3) dtype float32
            - mask shape(768, 768) dtype float32

        Returns:
            Cropped and possibly flipped image and mask pairs.
        """
        available_strategies = {
            "random_crop": self.random_crop,
            "semantic_label_inpainting_pair": self.semantic_label_inpainting_pair,
            "semantic_label_copy_paste_pair": self.semantic_label_copy_paste_pair,
        }

        enabled_strategies = [key for key, value in self.strategies.items() if value]

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

            # Apply random flipping and rotating to reduce similarities
            helper_corner, helper_mask = self.apply_transformations(
                helper_corner,
                helper_mask,
                [HorizontalFlip(p=0.5), RandomRotate90(p=0.5)],
            )
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

        # Convert back to uint8
        base_corner = (
            cv2.normalize(
                base_corner,
                None,
                base_corner.max(),
                base_corner.min(),
                cv2.NORM_MINMAX,
                cv2.CV_8U,
            )
            if base_corner.dtype != np.uint8
            else base_corner
        )
        base_mask = (
            cv2.normalize(
                base_mask,
                None,
                base_mask.max(),
                base_mask.min(),
                cv2.NORM_MINMAX,
                cv2.CV_8U,
            )
            if base_mask.dtype != np.uint8
            else base_mask
        )
        helper_corner = (
            cv2.normalize(
                helper_corner,
                None,
                helper_corner.max(),
                helper_corner.min(),
                cv2.NORM_MINMAX,
                cv2.CV_8U,
            )
            if helper_corner.dtype != np.uint8
            else helper_corner
        )
        helper_mask = (
            cv2.normalize(
                helper_mask,
                None,
                helper_mask.max(),
                helper_mask.min(),
                cv2.NORM_MINMAX,
                cv2.CV_8U,
            )
            if helper_mask.dtype != np.uint8
            else helper_mask
        )

        # Apply selected strategy method
        base_corner, base_mask, helper_corner, helper_mask = strategy_method(
            base_corner, base_mask, helper_corner, helper_mask
        )

        y[field.MASK1] = base_mask
        y[field.VMASK2] = helper_mask

        x = np.concatenate([base_corner, helper_corner], axis=2)

        return x, y

    def apply_transformations(self, image, mask, transformations):
        """
        applies transformations to the image and mask

        image, mask dtype : float32
        """
        composed_transform = Compose(transformations)
        augmented = composed_transform(image=image, mask=mask)

        return augmented["image"], augmented["mask"]

    def corner_selection(self, img, mask, mid_w, mid_h, strategy):
        img = cv2.normalize(
            img,
            None,
            img.max(),
            img.min(),
            cv2.NORM_MINMAX,
            cv2.CV_8U,
        )
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
        # Additional logic based on selected strategy
        if strategy == "semantic_label_inpainting_pair":
            base_corner_key, helper_corner_key = self.non_empty_corners(corners)

        elif strategy == "semantic_label_copy_paste_pair":
            helper_corner_key, base_corner_key = self.non_empty_corners(corners)

        else:
            base_corner_key, helper_corner_key = self.non_empty_corners(corners)

        # Obtain base corner and helper corner based on the selected keys
        base_corner, base_mask = corners[base_corner_key]
        helper_corner, helper_mask = corners[helper_corner_key]

        ###################################################################
        # img_mean = np.mean(img, axis=(0, 1))
        # img_std = np.std(img, axis=(0, 1))

        # is_similar_magnitude_base = self.check_scale_after_splitting(
        #     img_mean,
        #     img_std,
        #     corners[base_corner_key][0],
        # )
        # is_similar_magnitude_helper = self.check_scale_after_splitting(
        #     img_mean,
        #     img_std,
        #     corners[helper_corner_key][0],
        # )
        # if not is_similar_magnitude_base or not is_similar_magnitude_helper:
        #     print("Scale not similar after splitting")
        ###################################################################

        return (
            base_corner.astype(np.float32),
            base_mask,
            helper_corner.astype(np.float32),
            helper_mask,
        )

    def non_empty_corners(self, corners):
        non_empty_corner = [
            key for key, (corner, mask) in corners.items() if np.sum(mask) > 0
        ]

        if non_empty_corner:
            base_corner_key = random.choice(non_empty_corner)
        else:
            base_corner_key = random.choice(list(corners.keys()))

        corners = {
            key: value for key, value in corners.items() if key != base_corner_key
        }
        non_empty_corner = [key for key in non_empty_corner if key != base_corner_key]

        if non_empty_corner:
            helper_corner_key = random.choice(non_empty_corner)
        else:
            helper_corner_key = random.choice(list(corners.keys()))

        return base_corner_key, helper_corner_key

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
            inpaint_mask.astype(np.uint8),
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

    def fourier_blending(self, source_image, target_image, L=0.0005):
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

        # Apply FDA_source_to_target_np (input float32)
        src_in_tar = FDA_source_to_target_np(source_image, target_image, L=L)
        # Output float64

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

        if np.any((src_in_tar > 255) | (src_in_tar < 0)):
            raise ValueError(
                f"Expected no overflows but got {np.max(src_in_tar)}{np.min(src_in_tar)}"
            )
        return src_in_tar

    ### Helper methods for data augmentation
    def check_data(
        self,
        x,
        expected_dtype,
        expected_shape,
    ):
        """dtype, channel mean/max/min, overflows, geometric alignment"""
        if x.dtype != expected_dtype:
            raise ValueError(f"Expected dtype {expected_dtype} but got {x.dtype}")
        if x.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape} but got {x.shape}")
        if expected_dtype == np.uint8:
            if np.any((x > 255) | (x < 0)):
                raise ValueError(
                    f"Expected no overflows but got {np.max(x)}{np.min(x)}"
                )

    def check_scale_after_splitting(
        self, original_mean, original_std, corner_data, threshold=0.3
    ):
        corner_mean = np.mean(corner_data, axis=(0, 1))
        corner_std = np.std(corner_data, axis=(0, 1))

        # Check if absolute difference in mean and standard deviation is within threshold of original values
        is_similar_magnitude = np.all(
            abs(original_mean - corner_mean) < threshold * original_mean
        ) and np.all(abs(original_std - corner_std) < threshold * original_std)

        return is_similar_magnitude
