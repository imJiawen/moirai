import re
import torch
import torch.nn as nn

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
        if mask is not None:
            x = x * mask  
        return x

    def _denormalize(self, x, mask):
        x = x * self.stdev
        x = x + self.mean
        if mask is not None:
            x = x * mask
        return x
