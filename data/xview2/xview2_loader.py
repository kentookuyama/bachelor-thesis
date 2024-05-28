import random

import ever as er
import numpy
import torch
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    OneOf,
    RandomRotate90,
    VerticalFlip,
)
from torch.utils.data import ConcatDataset, SequentialSampler

from core.dataset import ColorAugDataset
from data.xview2.xview2_dataset import PreCachedXview2Building


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


@er.registry.DATALOADER.register()
class PreCachedXview2BuildingLoader(er.ERDataLoader):
    def __init__(self, config):
        super().__init__(config)

    @property
    def dataloader_params(self):
        if self.config.training:
            transform = None
        else:
            transform = self.config.common_transforms

        ## 1. self-pair generated (two 512x512 images)
        if isinstance(self.config.image_dir, (tuple, list)):
            dataset_list = []
            for im_dir, target_dir in zip(
                self.config.image_dir, self.config.target_dir
            ):
                dataset_list.append(
                    PreCachedXview2Building(
                        im_dir, target_dir, transform, self.config.strategies
                    )
                )

            dataset = ConcatDataset(dataset_list)

        else:
            dataset = PreCachedXview2Building(
                self.config.image_dir, self.config.target_dir, transform
            )

        ## 2.1 Data Augmentatioons (Geometric Transforms, RandomDiscreteScale)
        ## 2.2 Common Transforms (Normalization)
        ## 2.3 Convert to Tensor
        if self.config.training:
            dataset = ColorAugDataset(
                dataset,
                geo_transform=self.config.geo_transforms,
                color_transform=self.config.color_transforms,
                common_transform=self.config.common_transforms,
            )

        print(f"Dataloader length : {len(dataset)}")

        print(f"Dataloader length after xview2 : {len(dataset)}")

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
        print("done")
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
