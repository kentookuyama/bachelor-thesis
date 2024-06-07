import numpy
import torch
from torch.utils.data import Dataset

from core import field
from core.customtensor import CustomTensor


class ColorAugDataset(Dataset):
    def __init__(self, dataset, geo_transform, color_transform, common_transform):
        self.dataset = dataset
        self.color_transform = color_transform
        self.geo_transform = geo_transform
        self.common_transform = common_transform

    # def __init__(
    #     self, image_dir, targets_dir, geo_transform, color_transform, common_transform
    # ):
    #     self.target_fps = sorted(
    #         [fp for fp in glob.glob(os.path.join(targets_dir, "*.png")) if "pre" in fp]
    #     )

    #     self.image_fps = [
    #         os.path.join(image_dir, os.path.basename(fp.replace("_target.png", ".png")))
    #         for fp in self.target_fps
    #     ]
    #     self.color_transform = color_transform
    #     self.geo_transform = geo_transform
    #     self.common_transform = common_transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        if x.shape[2] == 6:
            # Extract corners and their respective masks from x and y
            base_corner_np = x[:, :, :3]
            helper_corner_np = x[:, :, 3:]
            mask1_np = y[field.MASK1]
            mask2_np = y[field.VMASK2]

            orig_shapes = {
                "base_corner": base_corner_np.shape,
                "helper_corner": helper_corner_np.shape,
                "mask1": mask1_np.shape,
                "mask2": mask2_np.shape,
            }

            if self.geo_transform:
                # Apply Geo Transforms
                image_1 = self.geo_transform(
                    **dict(image=base_corner_np, mask=mask1_np)
                )
                image_2 = self.geo_transform(
                    **dict(image=helper_corner_np, mask=mask2_np)
                )

                # Verify shapes after geo_transform
                if (
                    image_1["image"].shape != orig_shapes["base_corner"]
                    or image_1["mask"].shape != orig_shapes["mask1"]
                ):
                    raise ValueError(
                        f"Shape mismatch after geo_transform for base_corner or mask1: expected {orig_shapes['base_corner']}, {orig_shapes['mask1']} but got {image_1['image'].shape}, {image_1['mask'].shape}"
                    )

                if (
                    image_2["image"].shape != orig_shapes["helper_corner"]
                    or image_2["mask"].shape != orig_shapes["mask2"]
                ):
                    raise ValueError(
                        f"Shape mismatch after geo_transform for helper_corner or mask2: expected {orig_shapes['helper_corner']}, {orig_shapes['mask2']} but got {image_2['image'].shape}, {image_2['mask'].shape}"
                    )

                # Extract transformed images and masks
                image_1_img = image_1["image"]
                image_1_mask = image_1["mask"]
                image_2_img = image_2["image"]
                image_2_mask = image_2["mask"]
            else:
                image_1_img, image_1_mask, image_2_img, image_2_mask = (
                    base_corner_np,
                    mask1_np,
                    helper_corner_np,
                    mask2_np,
                )

            # uint8 (512,512,3) npdarray

            # x, mask -> tensor (common Transforms)
            image_1 = self.common_transform(image=image_1_img, mask=image_1_mask)
            image_2 = self.common_transform(image=image_2_img, mask=image_2_mask)

            image_1_img = image_1["image"]
            image_1_mask = image_1["mask"]
            image_2_img = image_2["image"]
            image_2_mask = image_2["mask"]

            # Convert to tensor
            to_tensor = CustomTensor()
            image_1 = to_tensor(image=image_1_img, mask=image_1_mask)
            image_2 = to_tensor(image=image_2_img, mask=image_2_mask)

            image_1_img = image_1["image"]
            image_1_mask = image_1["mask"]
            image_2_img = image_2["image"]
            image_2_mask = image_2["mask"]

            if (
                image_1_img.dtype != torch.float32
                or image_1_mask.dtype != torch.float32
            ):
                raise TypeError("Expected image and mask to be float32")

            # float32 (3,512,512) tensor

            # Ensure both images|masks are tensors
            if (
                not isinstance(image_1_img, torch.Tensor)
                or not isinstance(image_1_mask, torch.Tensor)
                or not isinstance(image_2_img, torch.Tensor)
                or not isinstance(image_2_mask, torch.Tensor)
            ):
                raise TypeError("Expected x and y to be PyTorch tensors")

            # Set data to dataset form required for ChangeMixin
            org_img = torch.cat([image_1_img, image_2_img], dim=0)
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
