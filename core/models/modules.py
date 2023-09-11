import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, pad_mode="reflect", **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get("dilation", 1)
        stride = kwargs.get("stride", 1)
        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0), mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        out = out[..., : (n * self.upsample_factor)]

        return out


def ResidualUnit(
    chan_in, chan_out, dilation, kernel_size=7, squeeze_excite=False, pad_mode="reflect"
):
    return Residual(
        Sequential(
            CausalConv1d(
                chan_in, chan_out, kernel_size, dilation=dilation, pad_mode=pad_mode
            ),
            nn.ELU(),
            CausalConv1d(chan_out, chan_out, 1, pad_mode=pad_mode),
            nn.ELU(),
            SqueezeExcite(chan_out) if squeeze_excite else None,
        )
    )
