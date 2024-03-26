import numpy as np
from torch.utils.data import Dataset

from core import field


class ColorAugDataset(Dataset):
    def __init__(self, dataset, geo_transform, color_transform, common_transform):
        print("ColorAugDatasSet initialized")
        self.dataset = dataset
        self.color_transform = color_transform
        self.geo_transform = geo_transform
        self.common_transform = common_transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = np.array(x)
        y["mask"] = np.array(y["mask"])

        try:
            blob = self.geo_transform(**dict(image=x, mask=y["mask"]))
        except ValueError as e:
            # Handle the case where the crop size exceeds the image size
            print(f"Skipping item due to crop size issue.")
            blob = dict(image=x, mask=y["mask"])
        img = blob['image']
        mask = blob['mask']
        print("after geo_transform")
        for e in img:
            print(f"geo_transform: {e.shape}\n")


        # x, mask -> tensor
        blob = self.common_transform(image=img, mask=mask)
        org_img = blob['image']
        mask = blob['mask']
        y["mask"] = mask
        print("after common_transform")
        for e in org_img:
            print(f"common_transform: {e.shape}\n")

        # x -> color_trans_x -> tensor
        if self.color_transform:
            color_trans_x = self.color_transform(**dict(image=img))['image']
            blob = self.common_transform(image=color_trans_x)
            color_trans_x = blob['image']
            y[field.COLOR_TRANS_X] = color_trans_x
            print("after color_transform")
            for e in color_trans_x:
                print(f"color_transform: {e.shape}\n")

        return org_img, y

    def __len__(self):
        return len(self.dataset)
