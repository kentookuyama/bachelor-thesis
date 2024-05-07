import numpy as np
import torch
from core import field
from torch.utils.data import Dataset


class ColorAugDataset(Dataset):
    def __init__(self, dataset, geo_transform, color_transform, common_transform):
        self.dataset = dataset
        self.color_transform = color_transform
        self.geo_transform = geo_transform
        self.common_transform = common_transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        if x.shape[1] == 99:
            base_corner_tensor = x[0, :, :]
            helper_corner_tensor = x[0, :, :]
            image_1 = self.geo_transform(
                **dict(image=base_corner_tensor.numpy(), mask=y[field.MASK1].numpy())
            )
            image_2 = self.geo_transform(
                **dict(
                    image=np.array(helper_corner_tensor), mask=np.array(y[field.VMASK2])
                )
            )
            image_1_img = image_1["image"]
            image_1_mask = image_1["mask"]
            image_2_img = image_2["image"]
            image_2_mask = image_2["mask"]

            print("here are the image shapes")
            print(image_1_img.shape)
            print(image_2_img.shape)
            print(image_1_mask.shape)
            print(image_2_mask.shape)

            # x, mask -> tensor
            image_1 = self.common_transform(image=image_1_img, mask=image_1_mask)
            image_2 = self.common_transform(image=image_2_img, mask=image_2_mask)

            image_1_img = image_1["image"]
            image_1_mask = image_1["mask"]
            image_2_img = image_2["image"]
            image_2_mask = image_2["mask"]

            org_img = torch.cat([image_1_img, image_2_img], dim=1)
            y[field.MASK1] = image_1_mask
            y[field.VMASK2] = image_2_mask

            return org_img, y

        blob = self.geo_transform(**dict(image=x, mask=y[field.MASK1]))
        img = blob["image"]
        mask = blob["mask"]

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
        return len(self.dataset)
