import glob
import os

import ever as er
import numpy as np
from albumentations import Compose, Normalize
from skimage.io import imread
from torch.utils.data import (
    ConcatDataset,
    Dataset,
    SequentialSampler,
    SubsetRandomSampler,
)


class LEVIRCD(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        print(f"LEVIRCD Dataset initialized with root_dir: {root_dir}")

        self.A_image_fps = glob.glob(os.path.join(root_dir, "A", "*.png"))
        self.B_image_fps = [fp.replace("A", "B") for fp in self.A_image_fps]
        self.gt_fps = [fp.replace("A", "label") for fp in self.A_image_fps]
        self.transforms = transforms

        print(f"Number of A images: {len(self.A_image_fps)} | root_dir: {root_dir}")
        print(f"Number of B images: {len(self.B_image_fps)} | root_dir: {root_dir}")
        print(
            f"Number of ground truth images: {len(self.gt_fps)} | root_dir: {root_dir}"
        )

    def __getitem__(self, idx):
        print(f"Loading item {idx + 1}/{len(self)}")

        img1 = imread(self.A_image_fps[idx])
        img2 = imread(self.B_image_fps[idx])
        gt = imread(self.gt_fps[idx])

        imgs = np.concatenate([img1, img2], axis=2)
        if self.transforms:
            blob = self.transforms(**dict(image=imgs, mask=gt))
            imgs = blob["image"]
            gt = blob["mask"]

        return imgs, dict(
            change=gt, image_filename=os.path.basename(self.A_image_fps[idx])
        )

    def __len__(self):
        length = len(self.A_image_fps)
        print(f"LEVIRCD Dataset size: {length} | root_dir: {self.root_dir}")
        return len(self.A_image_fps)


@er.registry.DATALOADER.register()
class LEVIRCDLoader(er.ERDataLoader):
    def __init__(self, config):
        print(f"LEVIRCDLoader initialized with config: {config}")
        super(LEVIRCDLoader, self).__init__(config)

    @property
    def dataloader_params(self):
        print(f"Config before checking root_dir: {self.config}")

        if isinstance(self.config.root_dir, (tuple, list)):

            dataset_list = []
            for im_dir in self.config.root_dir:
                dataset_list.append(LEVIRCD(im_dir, self.config.transforms))

            dataset = ConcatDataset(dataset_list)

        else:

            dataset = LEVIRCD(self.config.root_dir, self.config.transforms)

        if self.config.subsample_ratio < 1.0:
            num_train = len(dataset)
            indices = list(range(num_train))
            split = int(np.floor(self.config.subsample_ratio * num_train))
            sampler = SubsetRandomSampler(indices[:split])
        else:
            sampler = (
                er.data.StepDistributedSampler(dataset)
                if self.config.training
                else SequentialSampler(dataset)
            )

        print(f"LEVIRCDLoader dataset size: {len(dataset)}")

        return dict(
            dataset=dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
        )

    def set_default_config(self):
        self.config.update(
            dict(
                root_dir="",
                transforms=Compose(
                    [
                        Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                            max_pixel_value=255,
                        ),
                        er.preprocess.albu.ToTensor(),
                    ]
                ),
                subsample_ratio=0.1,
                batch_size=1,
                num_workers=0,
                training=False,
            )
        )
