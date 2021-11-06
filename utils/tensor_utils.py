import torch


# noinspection PyTypeChecker
def mask_pos0(tensor, pos_mask):
    """

    Args:
        tensor (torch.Tensor): (*)
        pos_mask (torch.Tensor): same size as logits
            1 for positions that are NOT MASKED, 0 for MASKED positions.

    Returns:
        torch.Tensor: same size as logits
    """
    if tensor.dtype == torch.float16:
        return tensor * pos_mask - 65500 * (1 - pos_mask)
    else:
        return tensor * pos_mask - 1e30 * (1 - pos_mask)


# noinspection PyTypeChecker
def mask_pos(x, mask):
    """

    Args:
        x (torch.Tensor):
        mask (torch.Tensor): same shape as x, 1 for available position, 0 for masked position

    Returns:
        torch.Tensor:
    """
    return x - 10000.0 * (1.0 - mask)


def pad_tensors(tensors, pad_val, left_pad=False, move_eos_to_beginning=False, eos_val=None):
    """Convert a list of 1d tensors into a padded 2d tensor."""

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_val
            dst[0] = eos_val
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if len(tensors[0].size()) > 1:
        tensors = [x.view(-1) for x in tensors]
    batch_size = len(tensors)
    max_len = max(x.size(0) for x in tensors)
    padded_tensor = tensors[0].new_full((batch_size, max_len), pad_val, requires_grad=tensors[0].requires_grad)
    for i, x in enumerate(tensors):
        copy_tensor(x, padded_tensor[i, max_len - len(x):] if left_pad else padded_tensor[i, :len(x)])
    return padded_tensor


def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.cuda()
    elif isinstance(obj, dict):
        return {k: to_cuda(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cuda(x) for x in obj]
    else:
        return obj


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(to_device(x, device) for x in obj)
    else:
        return obj
