"""A constant-resolution ConvNeXt predictor for NCHW fields."""

import math
from numbers import Integral

import torch
import torch.nn as nn

from networks.helpers import trunc_normal_
from networks.vit import MLP


def positive_int(value, name):
    """Return ``value`` when it is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def kernel_radius(kernel_size):
    """Return the halo width of an odd convolution kernel."""
    kernel_size = positive_int(kernel_size, "kernel_size")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    return (kernel_size - 1) // 2


def image_shape(img_size, patch_size):
    """Return a height and width that are divisible by ``patch_size``."""
    if not isinstance(img_size, (tuple, list)) or len(img_size) != 2:
        raise ValueError("img_size must be a (height, width) pair")
    shape = tuple(positive_int(v, "img_size") for v in img_size)
    if any(v % patch_size for v in shape):
        raise ValueError("img_size must be divisible by patch_size")
    return shape


class LayerNorm2d(nn.Module):
    """Normalize channels independently at each spatial location."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).to(x.dtype).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """Depthwise convolution and channel MLP with a residual connection."""

    def __init__(self, channels, kernel_size=7, layer_scale_init=1e-6):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size,
            padding=kernel_radius(kernel_size), groups=channels,
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = MLP(channels, 4 * channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), float(layer_scale_init)))

    def forward(self, x):
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        x = self.mlp(self.norm(x).to(x.dtype))
        x = (x * self.gamma.to(x.dtype)).permute(0, 3, 1, 2)
        return residual + x


class ConvNeXtField(nn.Module):
    """Constant-resolution ConvNeXt mapping an NCHW field to an NCHW field."""

    def __init__(
        self,
        img_size,
        in_chans=2,
        out_chans=2,
        patch_size=4,
        embed_dim=64,
        depth=8,
        kernel_size=7,
        layer_scale_init=1e-6,
    ):
        super().__init__()
        self.patch_size = positive_int(patch_size, "patch_size")
        self.img_size = image_shape(img_size, self.patch_size)
        self.input_shape = self.img_size
        self.embed_dim = positive_int(embed_dim, "embed_dim")
        self.out_ch = positive_int(out_chans, "out_chans")
        self.kernel_size = kernel_size
        self.halo = kernel_radius(kernel_size)
        in_chans = positive_int(in_chans, "in_chans")
        depth = positive_int(depth, "depth")
        if not math.isfinite(layer_scale_init) or layer_scale_init < 0:
            raise ValueError("layer_scale_init must be finite and nonnegative")
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size),
            LayerNorm2d(embed_dim),
        )
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(embed_dim, kernel_size, layer_scale_init)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, out_chans * patch_size * patch_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_head(self, x):
        """Fold patch tokens back into an NCHW field."""
        batch, height, width, _ = x.shape
        x = self.head(x).reshape(
            batch, height, width, self.patch_size, self.patch_size, self.out_ch,
        )
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(
            batch, self.out_ch, height * self.patch_size, width * self.patch_size,
        )

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[-2:]) != self.input_shape:
            raise ValueError(f"ConvNeXt expected NCHW input with spatial shape {self.input_shape}")
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 3, 1)
        return self.forward_head(self.norm(x).to(x.dtype))


def model_kwargs(params):
    """Read ConvNeXt constructor arguments from a parameter object."""
    return dict(
        img_size=tuple(params.img_size),
        in_chans=params.n_in_channels,
        out_chans=params.n_out_channels,
        patch_size=params.patch_size,
        embed_dim=params.embed_dim,
        depth=params.depth,
        kernel_size=params.get("kernel_size", 7),
        layer_scale_init=params.get("layer_scale_init", 1e-6),
    )


def ConvNeXt(params):
    """Build a ConvNeXtField from a parameter object."""
    return ConvNeXtField(**model_kwargs(params))
