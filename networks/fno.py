import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.helpers import _truncate_modes, _pad_modes, _zero_self_conjugate_imag


class FourierNeuralOperatorLayer(nn.Module):
    """
    Fourier neural operator layer: 2D real FFT, a complex channel-mixing weight per
    retained wavenumber (kh, kw), and the inverse FFT.

    Input and output are (B, channels, H, W) fields on the same grid. The FNO runs channels-first,
    so the FFTs act on the contiguous trailing dims. The pointwise layers are the ViT's MLP layers
    in their ``layout="channels_first"`` mode, see distributed/fft.py for the reasoning.

    ``modes_kh`` and ``modes_kw`` set the number of retained wavenumbers after truncation
    (defaults: all), and the weight has shape (in_channels, out_channels, modes_kh, modes_kw).
    Along h the retained modes are the lowest positive and negative frequencies; along w they
    are the lowest non-negative ones.

    The inverse transform is an ifft along h followed by a real ifft along w. Before the latter,
    the imaginary parts of the self-conjugate w columns (DC and Nyquist) are explicitly zeroed,
    since a c2r transform is only defined for them being real. This matches ``irfft2`` on CPU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        modes_kh: Optional[int] = None,
        modes_kw: Optional[int] = None,
        gain: float = 1.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.modes_kh = min(modes_kh or height, height)
        self.modes_kw = min(modes_kw or width // 2 + 1, width // 2 + 1)

        scale = math.sqrt(gain / in_channels)
        init = scale * torch.randn(
            in_channels,
            out_channels,
            self.modes_kh,
            self.modes_kw,
            dtype=torch.complex64,
        )
        self.weight = nn.Parameter(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = torch.fft.rfft2(x.to(torch.float32), s=(self.height, self.width), norm="ortho")
            x = _truncate_modes(x.narrow(-1, 0, self.modes_kw), self.modes_kh, dim=-2)

            x = torch.einsum("bihw,iohw->bohw", x, self.weight)

            # zero-fill the truncated modes, then the inverse real FFT
            x = _pad_modes(x, self.modes_kh, self.height, dim=-2)
            x = F.pad(x, (0, self.width // 2 + 1 - self.modes_kw))
            x = torch.fft.ifft(x, n=self.height, dim=-2, norm="ortho")
            x = _zero_self_conjugate_imag(x, self.width, dim=-1)
            x = torch.fft.irfft(x, n=self.width, dim=-1, norm="ortho")

        return x.to(dtype)


class MLP(nn.Module):
    """Pointwise MLP over the channels of a (B, C, H, W) field, as 1x1 convs."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """
    FNO block: ``x = act(spectral(x) + skip(x))``, followed by an optional residual channel MLP.
    The pointwise skip path carries the modes that the spectral layer truncates.
    """

    def __init__(self, dim, height, width, modes_kh=None, modes_kw=None, mlp_ratio=2.0, act_layer=nn.GELU):
        super().__init__()
        self.spectral = FourierNeuralOperatorLayer(dim, dim, height, width, modes_kh, modes_kw)
        self.skip = nn.Conv2d(dim, dim, 1)
        self.act = act_layer()
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, act_layer=act_layer) if mlp_ratio > 0 else None

    def forward(self, x):
        x = self.act(self.spectral(x) + self.skip(x))
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class FourierNeuralOperator(nn.Module):
    """
    Fourier neural operator: a pointwise lifting MLP to ``embed_dim`` channels, ``depth`` FNO
    blocks, and a pointwise projection MLP to ``out_chans``. Input and output are
    (B, C, H, W); all layers are channels-first, see distributed/fft.py.
    """

    def __init__(
        self,
        img_size=(256, 512),
        in_chans=4,
        out_chans=4,
        embed_dim=64,
        depth=4,
        modes_kh=None,
        modes_kw=None,
        mlp_ratio=2.0,
        **kwargs,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.lift = MLP(in_chans, embed_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, *self.img_size, modes_kh=modes_kh, modes_kw=modes_kw, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.project = MLP(embed_dim, embed_dim, out_chans)

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


def FNO(params, **kwargs):
    return FourierNeuralOperator(
        img_size=tuple(params.img_size),
        in_chans=params.n_in_channels,
        out_chans=params.n_out_channels,
        embed_dim=params.embed_dim,
        depth=params.depth,
        modes_kh=params.get("modes_kh", None),
        modes_kw=params.get("modes_kw", None),
        mlp_ratio=params.get("mlp_ratio", 2.0),
        **kwargs,
    )
