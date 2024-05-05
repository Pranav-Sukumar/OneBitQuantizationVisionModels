# Credits to Microsoft Paper: https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, feature_dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        # Ensure feature_dim is an integer, not a tensor
        self.scale = nn.Parameter(torch.ones((feature_dim,)))  # Add tuple packing
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


def activation_quant(x):
    """ Per-token quantization to 8 bits. No grouping is needed for quantization.
    Args:
        x: an activation tensor with shape [n, d]
    Returns:
        y: a quantized activation tensor with shape [n, d]
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return y

def weight_quant(w):
    """ Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.
    Args:
        w: a weight tensor with shape [d, k]
    Returns:
        u: a quantized weight tensor with shape [d, k]
    """
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    u = (w * scale).round().clamp(-1, 1) / scale
    return u

class BitLinear(nn.Linear):
    """ This is only for training, and kernel optimization is needed for efficiency.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.RMSNorm = RMSNorm(in_features)
    def forward(self, x):
        """ Args:
            x: an input tensor with shape [n, d]
        Returns:
            y: an output tensor with shape [n, d]
        """
        w = self.weight  # a weight tensor with shape [d, k]
        x_norm = self.RMSNorm(x)
        # A trick for implementing Straight-Through-Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y
