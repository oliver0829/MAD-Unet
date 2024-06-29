# -------------------------------------------------------------
# File: dice_score.py
# Author: Qiang Li
# Date of Completion: June 27, 2024
# Description: Loss Function, including diceloss and iouloss
# -------------------------------------------------------------
# Input/Output Information (IO):
# Input: /
# Output: /
# -------------------------------------------------------------

import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # print(input.size() == target.size(),input.size(), target.size())
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def iou_loss(input: Tensor, target: Tensor):
    pred = (input > 0.5).float()
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(-1, -2))
    union = (pred + target).clamp(0, 1).sum(dim=(-1, -2))
    # Calculate IoU
    iou = intersection / union
    # Calculate IoU loss
    iou_loss = 1 - iou.mean()
    return iou_loss

#
# def weighted_loss(input: Tensor, label: Tensor):
#     balanced_w = 1.1
#     prediction = input.float()
#     with torch.no_grad():
#         mask = label.clone()
#         num_positive = torch.sum((mask == 1).float()).float()
#         num_negative = torch.sum((mask == 0).float()).float()
#         beta = num_negative / (num_positive + num_negative)
#         mask[mask == 1] = beta
#         mask[mask == 0] = balanced_w * (1 - beta)
#     prediction = torch.sigmoid(prediction)
#     cost = torch.nn.functional.binary_cross_entropy(
#         prediction.float(), label.float(), weight=mask, reduction='none')
#     cost = torch.sum(cost.float().mean((1, 2,)))
#     return cost

