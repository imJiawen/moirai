import re
import torch
import torch.nn as nn
from typing import Optional, Dict

def convert_to_list(s):
    '''
    Convert prediction length strings into list
    e.g., '96-192-336-720' will be convert into [96,192,336,720]
    Input: str, list, int
    Returns: list
    '''
    if (type(s).__name__=='int'):
        return [s]
    elif (type(s).__name__=='list'):
        return s
    elif (type(s).__name__=='str'):
        elements = re.split(r'\D+', s)
        return list(map(int, elements))
    else:
        return None
    
class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        """
        :param eps: a value added for numerical stability
        """
        super(InstanceNorm, self).__init__()
        self.eps = eps

    def forward(self, x, mode:str, mask=None):
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == 'denorm':
            x = self._denormalize(x, mask)
        else: raise NotImplementedError
        return x

    def _get_statistics(self, x, mask):
        if mask is not None:
            observed_x = x * mask
            count = mask.sum(dim=tuple(range(1, x.ndim - 1)), keepdim=True)
            self.mean = (observed_x.sum(dim=tuple(range(1, x.ndim - 1)), keepdim=True) / count).detach()
            self.stdev = (torch.sqrt((observed_x - self.mean).pow(2).sum(dim=tuple(range(1, x.ndim - 1)), keepdim=True) / count + self.eps)).detach()
        else:
            dim2reduce = tuple(range(1, x.ndim-1))
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x, mask):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x, mask):
        x = x * self.stdev
        x = x + self.mean
        return x


def weighted_average(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: int = None,
    reduce: str = 'mean',
):
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        x: Input tensor, of which the average must be computed.
        weights: Weights tensor, of the same shape as `x`.
        dim: The dim along which to average `x`

    Returns:
        Tensor: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        if reduce != 'mean':
            return weighted_tensor
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim) if dim else x