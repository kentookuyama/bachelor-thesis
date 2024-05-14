import random
import sys

import cv2
import ever as er
import numpy as np
import torch
import torch.nn as nn
from core import field, loss
from core.head import get_detector
from FDA.utils import FDA_source_to_target_np

MAX_TIMES = 50


def generate_target(x1, y, strategies):
    # x [N, C * 1, H, W]
    # y dict(mask1=tensor[N, H, W], ...)
    original_image = x1
    original_mask = y[field.MASK1]
    ##################
    # N = x1.size(0)
    # org_inds = np.arange(N)
    # t = 0
    # while True and t <= MAX_TIMES:
    #     t += 1
    #     shuffle_inds = org_inds.copy()
    #     np.random.shuffle(shuffle_inds)

    #     ok = org_inds == shuffle_inds
    #     if all(~ok):
    #         break
    #################
    strategy_list = [
        strategy_name for strategy_name, enabled in strategies.items() if enabled
    ]
    print(strategy_list)
    ### Create array containing strategies that will be used
    base_corner, base_mask, helper_corner, helper_mask = (
        original_method(original_image, original_mask)
        if not strategies
        else split_image(original_image, original_mask, strategy_list)
    )

    x = torch.cat([base_corner, helper_corner], dim=1)
    y[field.MASK1] = base_mask
    y[field.VMASK2] = helper_mask
    return x, y


def generate_pseudo_pair(x1, y):
    # x [N, C * 1, H, W]
    # y dict(mask1=tensor[N, H, W], ...)
    mask1 = y[field.MASK1]
    N = x1.size(0)
    org_inds = np.arange(N)
    t = 0
    while True and t <= MAX_TIMES:
        t += 1
        shuffle_inds = org_inds.copy()
        np.random.shuffle(shuffle_inds)

        ok = org_inds == shuffle_inds
        if all(~ok):
            break
    virtual_x2 = x1[shuffle_inds, :, :, :]
    virtual_mask2 = mask1[shuffle_inds, ...]
    x = torch.cat([x1, virtual_x2], dim=1)

    y[field.VMASK2] = virtual_mask2
    return x, y


### original Method ChangeStar
def original_method(x1, y):
    N = x1.size(0)
    org_inds = np.arange(N)
    t = 0
    while True and t <= MAX_TIMES:
        t += 1
        shuffle_inds = org_inds.copy()
        np.random.shuffle(shuffle_inds)

        ok = org_inds == shuffle_inds
        if all(~ok):
            break
    virtual_x2 = x1[shuffle_inds, :, :, :]
    virtual_mask2 = y[shuffle_inds, ...]
    return x1, y, virtual_x2, virtual_mask2


def split_image(original_image, original_mask, strategy_list):
    img = original_image
    mask = original_mask
    N, _, h, w = img.shape  # Get the shape directly from the input tensor
    mid_h, mid_w = h // 2, w // 2

    # Corners: a=top-left, b=top-right, c=bottom-left, d=bottom-right
    corner_a = img[:, :, :mid_h, :mid_w]
    corner_b = img[:, :, :mid_h, mid_w:]
    corner_c = img[:, :, mid_h:, :mid_w]
    corner_d = img[:, :, mid_h:, mid_w:]

    # Masks for each corner
    mask_a = mask[:, :mid_h, :mid_w]
    mask_b = mask[:, :mid_h, mid_w:]
    mask_c = mask[:, mid_h:, :mid_w]
    mask_d = mask[:, mid_h:, mid_w:]

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

    print(strategy_list)
    # Apply self-pairing strategy
    return getattr(sys.modules[__name__], random.choice(strategy_list))(
        base_corner, base_mask, helper_corner, helper_mask
    )


############ Strategies


##### Strategy 1: Random Crop
def random_crop(base_corner, base_mask, helper_corner, helper_mask):
    return base_corner, base_mask, helper_corner, helper_mask


##### Strategy 2: Inpainting
def semantic_label_inpainting_pair(base_corner, base_mask, helper_corner, helper_mask):
    mask = np.array(base_mask.cpu())
    print(mask.shape)
    # Find connected components in the mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=4
    )

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

    # Randomly select objects to inpaint from eligible objects
    num_objects_to_remove = np.random.randint(1, len(eligible_objects) + 1)
    selected_object_indices = np.random.choice(
        eligible_objects, num_objects_to_remove, replace=False
    )

    # Create a binary mask for inpainting
    inpaint_mask = np.zeros_like(mask)
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
    updated_mask = np.where(inpaint_mask == 1, 0, mask)
    inpainted_corner = torch.from_numpy(inpainted_corner)
    updated_mask = torch.from_numpy(updated_mask)

    return base_corner, base_mask, inpainted_corner, updated_mask


##### Strategy 3: copy & paste + Fourier blending
def semantic_label_copy_paste_pair(base_corner, base_mask, helper_corner, helper_mask):
    # Get the indices of non-background pixels in the helper mask
    helper_indices = np.argwhere(helper_mask != 0)

    # Iterate over each non-background pixel in the helper mask
    for idx in helper_indices:
        x, y = idx

        # If the pixel is within the bounds of the base mask, copy the corresponding pixel from the helper corner
        if 0 <= x < base_mask.shape[0] and 0 <= y < base_mask.shape[1]:
            base_corner[x, y, :] = helper_corner[x, y, :]
            base_mask[x, y] = helper_mask[x, y]

    # Apply Fourier transform based blending
    blended_corner = fourier_blending(base_corner, helper_corner)

    # Adjust the helper mask accordingly
    blended_mask = base_mask.copy()

    return base_corner, base_mask, blended_corner, blended_mask


def fourier_blending(source_image, target_image, L=0.01):
    # Ensure source and target images are in the same format
    source_image = source_image.transpose((2, 0, 1))
    target_image = target_image.transpose((2, 0, 1))

    # Apply FDA_source_to_target_np
    src_in_tar = FDA_source_to_target_np(source_image, target_image, L=L)

    # Transpose back to (H, W, C) format
    src_in_tar = src_in_tar.transpose((1, 2, 0))

    return src_in_tar


class ChangeMixin(nn.Module):
    def __init__(self, feature_extractor, classifier, detector_config, loss_config):
        super(ChangeMixin, self).__init__()
        self.features = feature_extractor
        self.classifier = classifier
        self.detector_config = detector_config
        self.change_detector = get_detector(**detector_config)
        self.loss_config = er.config.AttrDict.from_dict(loss_config)

    def extract_feature(self, x):
        return self.features(x)

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x, y=None):
        if self.training:
            # HERE IS WHERE PSEUDO PAIR IS CREATED!!
            if x.size(1) == 99:
                print("target generated")
                x, y = generate_pseudo_pair(x, y, self.detector_config.strategies)

            x1 = x[:, :3, :, :]
            vx2 = x[:, 3:, :, :]

            y1_feature = self.extract_feature(x1)
            vy2_feature = self.extract_feature(vx2)

            y1_pred = self.classify(y1_feature)

            # extract positive feature
            if self.detector_config.get("t1t2", True):
                change_y1vy2_logit = self.change_detector(
                    torch.cat([y1_feature, vy2_feature], dim=1)
                )
            else:
                change_y1vy2_logit = None
            if self.detector_config.get("t2t1", True):
                change_y2vy1_logit = self.change_detector(
                    torch.cat([vy2_feature, y1_feature], dim=1)
                )
            else:
                change_y2vy1_logit = None

            y1_true = y[field.MASK1]
            vy2_true = y[field.VMASK2]

            loss_dict = dict()
            loss_dict.update(loss.misc_info(y1_pred.device))

            if self.detector_config.get("symmetry_loss", False):
                loss_dict.update(
                    loss.semantic_and_symmetry_loss(
                        y1_true,
                        vy2_true,
                        y1_pred,
                        change_y1vy2_logit,
                        change_y2vy1_logit,
                        self.loss_config,
                    )
                )
            else:
                raise ValueError()

            return loss_dict

        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]

        y1_feature = self.extract_feature(x1)
        y2_feature = self.extract_feature(x2)

        y1_pred = self.classify(y1_feature)

        change_y1y2_logit = self.change_detector(
            torch.cat([y1_feature, y2_feature], dim=1)
        )

        y2_pred = self.classify(y2_feature)

        logit1 = y1_pred
        logit2 = y2_pred
        change_logit = change_y1y2_logit
        return torch.cat([logit1, logit2, change_logit], dim=1)
