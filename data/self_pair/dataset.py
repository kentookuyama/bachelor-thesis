import glob
import os
import random

import cv2
import ever as er
import numpy as np
import torch
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    OneOf,
    RandomCrop,
    RandomRotate90,
    VerticalFlip,
)
from PIL import Image
from skimage.io import imread
from torch.utils.data import ConcatDataset, Dataset, SequentialSampler

from core.dataset import ColorAugDataset
from FDA.utils import FDA_source_to_target_np


class MyCustomDataset(Dataset):
    def __init__(self, dataset, strategy_prob):
        print("MyCustomDataset initialized")
        self.dataset = dataset
        self.strategy_prob = strategy_prob
        
        # Load necessary strategies
        strategies = [
            self.random_crop_pair,
            self.semantic_label_inpainting_pair,
            self.semantic_label_copy_paste_pair,
        ]
        strategy_probs_array = np.array(list(strategy_prob.values()))
        weights = strategy_probs_array/sum(strategy_probs_array)
        self.self_pairing_methods = np.random.choice(strategies, p=weights)
        print(f"Currently Activate pairing methods | {self.self_pairing_methods}")

        print(
            f"Number of images: {len(self.image_fps)} | self.root_dir: {self.root_dir}"
        )
        print(
            f"Number of masks: {len(self.target_fps)} | self.root_dir: {self.root_dir}"
        )

    ### Self-pair strategies

    ##### Strategy 1: Random Crop
    def random_crop_pair(self, base_corner, base_mask, helper_corner, helper_mask):
        return base_corner, base_mask, helper_corner, helper_mask

    ##### Strategy 2: Inpainting
    def semantic_label_inpainting_pair(
        self, base_corner, base_mask, helper_corner, helper_mask
    ):
        # Assume that the mask has binary values, where 1 represents the objects to be erased
        objects_to_erase = np.where(base_mask == 1)
        # Randomly select some instances to erase
        num_instances_to_erase = random.randint(1, len(objects_to_erase[0]))
        selected_indices = random.sample(
            range(len(objects_to_erase[0])), num_instances_to_erase
        )

        # Create a binary mask (a) for inpainting
        a = np.zeros_like(base_mask)
        for idx in selected_indices:
            a[objects_to_erase[0][idx], objects_to_erase[1][idx]] = 1

        # Inpainting using OpenCV's inpaint method from TELEA as suggested from the paper
        # https://docs.opencv.org/3.4/d7/d8b/group__photo__inpaint.html#gga8c5f15883bd34d2537cb56526df2b5d6a892824c38e258feb5e72f308a358d52e
        inpainted_helper_corner = cv2.inpaint(
            base_corner, a, inpaintRadius=3, flags=cv2.INPAINT_TELEA
        )

        inpainted_helper_mask = np.where(a == 1, 0, base_mask)

        return base_corner, base_mask, inpainted_helper_corner, inpainted_helper_mask

    ##### Strategy 3: copy & paste + Fourier blending
    def semantic_label_copy_paste_pair(
        self, base_corner, base_mask, helper_corner, helper_mask
    ):
        org_corner, org_mask = base_corner, base_mask
        # Print relevant information
        print(
            f"Base corner shape: {base_corner.shape} | Base mask shape: {base_mask.shape}"
        )
        print(
            f"Helper corner shape: {helper_corner.shape} | Helper mask shape: {helper_mask.shape}"
        )
        # Copy the helper_corner onto the base_corner
        base_corner[:, :, : helper_corner.shape[2]] = helper_corner

        # Copy the helper_mask onto the base_mask
        base_mask[:, :] = helper_mask

        # Apply Fourier transform based blending
        blended_corner = self.fourier_blending(base_corner, helper_corner)

        print(f"Blended corner shape: {blended_corner.shape}")

        # Adjust the helper_mask accordingly
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

    ### Applying all methods onto one image mask combination

    def apply_self_pairing(self, base_corner, base_mask, helper_corner, helper_mask):
        # Debug Print statement
        print("def apply_self_pairing")

        if self.self_pairing_methods:
            # Apply method on mask
            # Debug Print statement
            print(f"corner shape: {base_corner.shape} | mask contains 1: {'yes' if 1 in base_mask else 'no'}")
            base_corner, base_mask, helper_corner, helper_mask = self.random_crop_pair(
                base_corner, base_mask, helper_corner, helper_mask
            )
        return helper_corner, helper_mask

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        img = np.array(x)
        mask = np.array(y["mask"])

        try:
            # Split image into four equal squares (corners)
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

            # Original image and mask
            original_img, original_mask = base_corner, base_mask    
            
            # Print information for debugging
            # print(f"Original Mask Shape: {original_mask.shape}")
            # for corner_key, corner_mask in corner_masks.items():
            #     print(f"Corner {corner_key} Mask Shape: {corner_mask.shape}")

            # Apply self-pairing strategy
            transformed_img, transformed_mask = self.apply_self_pairing(
                base_corner, base_mask, helper_corner, helper_mask
            )
            
            original_img = torch.from_numpy(original_img)
            transformed_img = torch.from_numpy(transformed_img)
            original_mask = torch.from_numpy(original_mask)
            transformed_mask = torch.from_numpy(transformed_mask)
            
            print("image shapes")
            print(f"original_img: {original_img.shape} | mask: {original_mask.shape}")
            print(f"transformed_mask: {transformed_img.shape} | mask: {transformed_mask.shape}")
            
            imgs = torch.cat([original_img, transformed_img])
            masks = torch.cat([original_mask, transformed_mask])

            y["mask"] = masks
            return (
                imgs,
                y,
            )
        except Exception as e:
            print(f"Error occurred in __getitem__ at index {idx}: {e}")
            raise

    def __len__(self):
        return len(self.dataset)


@er.registry.DATALOADER.register()
class MyCustomLoader(er.ERDataLoader):
    def __init__(self, config):
        # Debug Print statement
        print(f"MyCustomLoader initialized with config: {config}")
        super(MyCustomLoader, self).__init__(config)

    @property
    def dataloader_params(self):
        # Debug Print statement
        print(f"Config before checking image_dir: {self.config}")

        if self.config.training:
            transform = None
        else:
            transform = self.config.common_transforms
        
        if isinstance(self.config.image_dir, (tuple, list)):
            dataset_list = []
            for im_dir, target_dir in zip(
                self.config.image_dir, self.config.target_dir
            ):
                dataset_list.append(MyCustomDataset(im_dir, target_dir, self.config.strategy_prob, transform))

            dataset = ConcatDataset(dataset_list)
        else:
            dataset = MyCustomDataset(
                self.config.image_dir, self.config.target_dir, self.config.strategy_prob, transform
            )

        if self.config.training:
            print(f"Pre change size: {len(dataset)}")
            print("Dataset overridden?????")
            dataset = ColorAugDataset(
                dataset,
                geo_transform=self.config.geo_transforms,
                color_transform=self.config.color_transforms,
                common_transform=self.config.common_transforms,
            )
            print(f"Post change size: {len(dataset)}")

        if self.config.CV.on and self.config.CV.cur_k != -1:
            train_sampler, val_sampler = er.data.make_CVSamplers(
                dataset,
                cur_k=self.config.CV.cur_k,
                k_fold=self.config.CV.k_fold,
                distributed=True,
                seed=2333,
            )
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = (
                er.data.StepDistributedSampler(dataset)
                if self.config.training
                else SequentialSampler(dataset)
            )

        print(f"MyCustomLoader dataset size: {len(dataset)}")

        return dict(
            dataset=dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=0,
            worker_init_fn=seed_worker,
        )

    def set_default_config(self):
        self.config.update(
            dict(
                image_dir="",
                target_dir="",
                CV=dict(
                    on=True,
                    cur_k=0,
                    k_fold=5,
                ),
                geo_transforms=Compose(
                    [
                        OneOf(
                            [
                                HorizontalFlip(True),
                                VerticalFlip(True),
                                RandomRotate90(True),
                            ],
                            p=0.5,
                        ),
                        er.preprocess.albu.RandomDiscreteScale(
                            [0.75, 1.25, 1.5], p=0.5
                        ),
                        RandomCrop(512, 512, True),
                    ]
                ),
                color_transforms=None,
                common_transforms=Compose(
                    [
                        Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                            max_pixel_value=255,
                        ),
                        er.preprocess.albu.ToTensor(),
                    ]
                ),
                batch_size=1,
                num_workers=0,
                training=True,
            )
        )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
