import ever as er
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    OneOf,
    RandomCrop,
    RandomRotate90,
    VerticalFlip,
)

data = dict(
    train=dict(
        type="MyCustomLoader",
        params=dict(
            root_dir=("./LEVIR-CD/train",),
            transforms=Compose(
                [
                    Normalize(
                        mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225),
                        max_pixel_value=255,
                    ),
                    er.preprocess.albu.ToTensor(),
                ]
            ),
            batch_size=16,  # 16
            num_workers=8,  # 8
            training=True,
        ),
    ),
    test=dict(
        type="MyCustomLoader",
        params=dict(
            root_dir=("./LEVIR-CD/train", "./LEVIR-CD/val", "./LEVIR-CD/test"),
            transforms=Compose(
                [
                    Normalize(
                        mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225),
                        max_pixel_value=255,
                    ),
                    er.preprocess.albu.ToTensor(),
                ]
            ),
            batch_size=4,  # 4
            num_workers=2,  # 2
            training=False,
        ),
    ),
)
optimizer = dict(
    type="sgd",
    params=dict(momentum=0.9, weight_decay=0.0001),
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    ),
)
learning_rate = dict(
    type="poly",
    params=dict(
        base_lr=0.03 * (data["test"]["params"]["batch_size"] / 16),
        power=0.9,
        max_iters=40000,
    ),
)
train = dict(
    forward_times=1,
    num_iters=40000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=False,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=1000,
    eval_interval_epoch=10,
)
test = dict()
