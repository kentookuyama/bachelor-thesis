import ever.module as M
import numpy as np
import torch


def misc_info(device):
    loss_dict = dict()
    mem = torch.cuda.max_memory_allocated() // 1024 // 1024
    loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(device)
    return loss_dict


def bce_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor, loss_config, prefix=''):
    loss_dict = dict()
    weight = loss_config.get('weight', 1.0)
    if loss_config.get('dice', False):
        loss_dict[f'{prefix}dice_loss'] = weight * M.loss.dice_loss_with_logits(y_pred, y_true,
                                                                                ignore_index=loss_config.ignore_index)
    if loss_config.get('bce', False):
        loss_dict[f'{prefix}bce_loss'] = weight * M.loss.binary_cross_entropy_with_logits(y_pred, y_true,
                                                                                          ignore_index=loss_config.ignore_index)

    return loss_dict


def semantic_and_symmetry_loss(y1_true,
                               y2_true,
                               y1_pred,
                               y2_pred,
                               change_y1y2_logit,
                               change_y2y1_logit,
                               loss_config
                               ):
    # XOR mask between original and transformed image
    positive_mask = torch.logical_xor(y1_true, y2_true)

    total_loss = dict()
    # Changestar Self-pair should hit here
    if 'semantic' in loss_config and getattr(loss_config.semantic, 'on', True):
        total_loss.update(bce_dice_loss(y1_true, y1_pred, loss_config.semantic, 's'))
        total_loss.update(bce_dice_loss(y2_true, y2_pred, loss_config.semantic, 's'))
    if change_y1y2_logit is not None:
        total_loss.update(
            bce_dice_loss(positive_mask, change_y1y2_logit, loss_config.change, 'c12'))
    if change_y2y1_logit is not None:
        total_loss.update(
            bce_dice_loss(positive_mask, change_y2y1_logit, loss_config.change, 'c21'))

    return total_loss
