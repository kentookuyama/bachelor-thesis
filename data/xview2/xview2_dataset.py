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
        # img uint8 mask uint8
        x = cv2.cvtColor(cv2.imread(self.image_fps[idx]), cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32) / 255.0
        mask = cv2.imread(self.target_fps[idx], cv2.IMREAD_GRAYSCALE)
        y = {field.MASK1: mask, "image_filename": os.path.basename(self.image_fps[idx])}

        if self.strategies is None:
            return x, y

        if self.transforms:
            x, y[field.MASK1] = self.apply_transforms(x, y[field.MASK1])

        return self.split_image(x, y)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.image_fps)

    ### Methods for creating the dataset

    def apply_transforms(self, image, mask):
        """Applies the transforms to the image and mask"""
        image = image.astype(np.float32)
        augmented = self.transforms(**dict(image=image, mask=mask))
        return augmented["image"], augmented["mask"]

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
        # TODO TEMP DISABLED
        # base_corner = self.normalize_to_uint8(base_corner)
        # base_mask = self.normalize_to_uint8(base_mask)
        # helper_corner = self.normalize_to_uint8(helper_corner)
        # helper_mask = self.normalize_to_uint8(helper_mask)

        # Apply selected strategy method
        base_corner, base_mask, helper_corner, helper_mask = strategy_method(
            base_corner, base_mask, helper_corner, helper_mask
        )
        y[field.MASK1] = base_mask
        y[field.VMASK2] = helper_mask

        x = np.concatenate([base_corner, helper_corner], axis=2)
        return x, y

    def normalize_to_uint8(self, image):
        return (
            image
            if image.dtype == np.uint8
            else cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        )

    def apply_transformations(self, image, mask, transformations):
        """
        applies transformations to the image and mask

        image, mask dtype : float32
        """
        composed_transform = Compose(transformations)
        augmented = composed_transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]

    def corner_selection(self, img, mask, mid_w, mid_h, strategy):
        # img = cv2.normalize(
        #     img,
        #     None,
        #     img.max(),
        #     img.min(),
        #     cv2.NORM_MINMAX,
        #     cv2.CV_8U,
        # )
        # Corners: a=top-left, b=top-right, c=bottom-left, d=bottom-right
        corners = {
            "a": (img[:mid_h, :mid_w], mask[:mid_h, :mid_w]),
            "b": (img[:mid_h, -mid_w:], mask[:mid_h, -mid_w:]),
            "c": (img[-mid_h:, :mid_w], mask[-mid_h:, :mid_w]),
            "d": (img[-mid_h:, -mid_w:], mask[-mid_h:, -mid_w:]),
        }

        if strategy == "semantic_label_inpainting_pair":
            base_corner_key, helper_corner_key = self.non_empty_corners(corners)

        elif strategy == "semantic_label_copy_paste_pair":
            helper_corner_key, base_corner_key = self.non_empty_corners(corners)

        else:
            base_corner_key, helper_corner_key = self.non_empty_corners(corners)

        base_corner, base_mask = corners[base_corner_key]
        helper_corner, helper_mask = corners[helper_corner_key]

        return (
            base_corner,
            base_mask,
            helper_corner,
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
        eligible_objects, labels = self.eligible_image(base_mask)
        if not eligible_objects:
            self.skipped_amount += 1
            if self.skipped_amount % 200 == 0:
                print(f"Skipped {self.skipped_amount} times.")
            return self.random_crop(base_corner, base_mask, helper_corner, helper_mask)

        # Create Binary mask for inpainting eligible objects from the helper mask
        num_objects_to_remove = np.random.randint(1, len(eligible_objects) + 1)
        selected_object_indices = np.random.choice(
            eligible_objects, num_objects_to_remove, replace=False
        )

        inpaint_mask = np.zeros_like(base_mask, dtype=np.uint8)
        for index in selected_object_indices:
            inpaint_mask[labels == index + 1] = 1

        # Inpainting using OpenCV's inpaint method from TELEA
        inpainted_corner = cv2.inpaint(
            cv2.normalize(
                base_corner,
                None,
                255,
                0,
                cv2.NORM_MINMAX,
                cv2.CV_8U,
            ),
            inpaint_mask,
            20,
            cv2.INPAINT_TELEA,
        )

        updated_mask = np.where(inpaint_mask == 1, 0, base_mask)

        return (
            cv2.normalize(
                base_corner,
                None,
                255,
                0,
                cv2.NORM_MINMAX,
                cv2.CV_8U,
            ),
            base_mask,
            inpainted_corner,
            updated_mask,
        )

    ##### Strategy 3: copy & paste + Fourier blending
    def semantic_label_copy_paste_pair(
        self, base_corner, base_mask, helper_corner, helper_mask
    ):
        # Get eligible objects from the helper_corner
        eligible_objects, _ = self.eligible_image(helper_mask)

        # If object sizes are larger than 30% in image size and no objects are eligible, use random_crop
        if not eligible_objects:
            self.skipped_amount += 1
            if self.skipped_amount % 200 == 0:
                print(f"Skipped {self.skipped_amount} times.")
            return self.random_crop(base_corner, base_mask, helper_corner, helper_mask)

        final_corner = base_corner.copy()
        final_mask = base_mask.copy()
        ### COPY AND PASTE COMES HERE
        inpaint_mask = np.zeros_like(final_mask, dtype=np.uint8)
        for obj_index in eligible_objects:
            inpaint_mask[helper_mask == obj_index + 1] = 1

        # Vectorized approach to copy pixels
        final_corner[inpaint_mask == 1] = helper_corner[inpaint_mask == 1]
        final_mask[inpaint_mask == 1] = helper_mask[inpaint_mask == 1]

        base_corner_fda = self.fourier_blending(final_corner, helper_corner)

        return (
            cv2.normalize(
                base_corner,
                None,
                255,
                0,
                cv2.NORM_MINMAX,
                cv2.CV_8U,
            ),
            base_mask,
            base_corner_fda,
            final_mask,
        )

    def fourier_blending(self, source_image, target_image, L=0.0005):
        source_image = source_image.transpose((2, 0, 1))
        target_image = target_image.transpose((2, 0, 1))

        # Apply FDA_source_to_target_np (input float32)
        src_in_tar = FDA_source_to_target_np(source_image, target_image, L=L)
        # Output float64

        # Transpose back to (H, W, C) format
        src_in_tar = src_in_tar.transpose((1, 2, 0))

        # Convert Float64 to Uint8
        src_in_tar = cv2.normalize(
            src_in_tar,
            None,
            255,
            0,
            cv2.NORM_MINMAX,
            cv2.CV_8U,
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
        if expected_dtype == np.float32:
            if np.any((x > 1) | (x < 0)):
                raise ValueError(
                    f"Expected no overflows but got {np.max(x)}{np.min(x)}"
                )
