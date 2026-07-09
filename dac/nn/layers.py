import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for 1D convolutions.
    
    Applies RMS normalization over the channel dimension for 1D conv features.
    Input shape: [B, C, T] where B=batch, C=channels, T=time
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(math.sqrt(eps) * torch.ones(1, dim, 1))
        self.eps = eps
    
    def forward(self, x):
        # Normalize and scale
        return self.scale * x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)


class PaddedConv1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs.pop('padding', None)
        causal = kwargs.pop('causal', False)
        padding = (kwargs.get('kernel_size', 1) - 1) * kwargs.get('dilation', 1) - kwargs.get('stride', 1) + 1
        self.padding = (
            padding if causal else padding // 2,
            0 if causal else padding // 2
        )
        self.conv = weight_norm(nn.Conv1d(*args, **kwargs))
        
    def forward(self, x):
        # Pad only at the beginning
        x = F.pad(x, self.padding)
        return self.conv(x)

class PaddedConvTranspose1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs.pop('padding', None)
        causal = kwargs.pop('causal', False)
        padding = (kwargs.get('kernel_size', 1) - 1) * kwargs.get('dilation', 1) - kwargs.get('stride', 1) + 1
        self.trim_left = 0 if causal else padding // 2
        self.trim_right = padding if causal else padding // 2
        self.conv = weight_norm(nn.ConvTranspose1d(*args, **kwargs))
        
    def forward(self, x):
        x = self.conv(x)
        if self.trim_right > 0:
            return x[..., self.trim_left:-self.trim_right]
        return x[..., self.trim_left:]

def WNConv1d(*args, **kwargs):
    return PaddedConv1d(*args, **kwargs)

def WNConvTranspose1d(*args, **kwargs):
    return PaddedConvTranspose1d(*args, **kwargs)

# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)
