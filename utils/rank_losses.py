from itertools import product

import torch
from torch.nn import BCEWithLogitsLoss, MarginRankingLoss

from .tensor_utils import mask_pos

DEFAULT_EPS = 1e-10


def list_mle(y_pred, y_true, mask=None, reduction='mean', eps=DEFAULT_EPS):
    """ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".

    Args:
        y_pred (torch.FloatTensor): (N, L) predictions from the model
        y_true (torch.FloatTensor): (N, L) ground truth labels
        mask (torch.FloatTensor): (N, L) 1 for available position, 0 for masked position
        reduction: 'none' | 'mean' | 'sum'
        eps (float): epsilon value, used for numerical stability

    Returns:
        torch.Tensor: scalar if `reduction` is not 'none' else (N,)
    """
    # shuffle for randomized tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    shuffled_y_pred = y_pred[:, random_indices]
    shuffled_y_true = y_true[:, random_indices]
    shuffled_mask = mask[:, random_indices] if mask is not None else None

    sorted_y_true, rank_true = shuffled_y_true.sort(descending=True, dim=1)
    y_pred_in_true_order = shuffled_y_pred.gather(dim=1, index=rank_true)
    if shuffled_mask is not None:
        y_pred_in_true_order = mask_pos(y_pred_in_true_order, shuffled_mask)

    max_y_pred, _ = y_pred_in_true_order.max(dim=1, keepdim=True)
    y_pred_in_true_order_minus_max = y_pred_in_true_order - max_y_pred
    cum_sum = y_pred_in_true_order_minus_max.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    observation_loss = torch.log(cum_sum + eps) - y_pred_in_true_order_minus_max
    if shuffled_mask is not None:
        observation_loss[shuffled_mask == 0] = 0.0
    loss = observation_loss.sum(dim=1)

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


def list_net(y_pred, y_true, mask=None, irrelevant_val=None, reduction='mean'):
    """ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".

    Args:
        y_pred (torch.FloatTensor): (N, L) predictions from the model
        y_true (torch.FloatTensor): (N, L) ground truth labels
        mask (torch.FloatTensor): (N, L) 1 for available position, 0 for masked position
        irrelevant_val (float):
        reduction: 'none' | 'mean' | 'sum'

    Returns:
         torch.Tensor: scalar if `reduction` is not 'none' else (N,)
    """
    if mask is not None:
        y_pred = mask_pos(y_pred, mask)
        y_true = mask_pos(y_true, mask)
    if irrelevant_val is not None:
        y_true = mask_pos(y_true, (y_true != irrelevant_val).float())

    return soft_cross_entropy_with_logits(y_pred, y_true.softmax(dim=1), reduction)


def rank_net(y_pred, y_true, mask=None, weight_by_diff=False, weight_by_diff_powered=False, reduction='mean'):
    """RankNet loss introduced in "Learning to Rank using Gradient Descent".

    Args:
        y_pred (torch.FloatTensor): (N, L) predictions from the model
        y_true (torch.FloatTensor): (N, L) ground truth labels
        mask (torch.FloatTensor): (N, L) 1 for available position, 0 for masked position
        weight_by_diff: whether to weight the score differences by ground truth differences.
        weight_by_diff_powered: whether to weight the score differences by the squared ground truth differences
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        torch.Tensor: scalar if `reduction` is not 'none' else (N,)
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    if mask is not None:
        y_pred[mask == 0] = float('-inf')
        y_true[mask == 0] = float('-inf')

    # here we generate every pair of indices from the range of candidates number in the batch
    candidate_pairs = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, candidate_pairs]
    pairs_pred = y_pred[:, candidate_pairs]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symmetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pairs_pred[:, :, 0] - pairs_pred[:, :, 1]
    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powered:
        true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return BCEWithLogitsLoss(weight=weight, reduction=reduction)(pred_diffs, true_diffs)


def pairwise_hinge(y_pred, y_true, mask=None, reduction='mean'):
    """RankNet loss introduced in "Learning to Rank using Gradient Descent".

    Args:
        y_pred (torch.FloatTensor): (N, L) predictions from the model
        y_true (torch.FloatTensor): (N, L) ground truth labels
        mask (torch.FloatTensor): (N, L) 1 for available position, 0 for masked position
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        torch.Tensor: scalar if `reduction` is not 'none' else (N,)
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    if mask is not None:
        y_pred[mask == 0] = float('-inf')
        y_true[mask == 0] = float('-inf')

    # here we generate every pair of indices from the range of candidates number in the batch
    candidate_pairs = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, candidate_pairs]
    pairs_pred = y_pred[:, candidate_pairs]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symmetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    s1 = pairs_pred[:, :, 0][the_mask]
    s2 = pairs_pred[:, :, 1][the_mask]
    target = the_mask.float()[the_mask]

    return MarginRankingLoss(margin=1, reduction=reduction)(s1, s2, target)


def soft_cross_entropy_with_logits(logits, soft_labels, reduction='mean'):
    """

    Args:
        logits (torch.Tensor): (N, C)
        soft_labels (torch.Tensor): (N, C)
        reduction: 'none' | 'mean' | 'sum'

    Returns:
         torch.Tensor: scalar if `reduction` is not 'none' else (N,)
    """
    loss = -(soft_labels * logits.log_softmax(dim=1)).sum(dim=1)

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()
