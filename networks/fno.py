import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.helpers import (
    _truncate_modes,
    _pad_modes,
    _zero_self_conjugate_imag,
    resample_modes,
    spectral_init_decay,
    spectral_wavenumbers,
)


def init_pointwise_(weight, bias, fan_in):
    """
    Initialize a pointwise (1x1) channel-mixing layer: weights ``N(0, 1/(3 fan_in))``, biases
    ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``. This is the variance of PyTorch's default for
    ``nn.Linear`` / ``nn.Conv2d`` (kaiming-uniform with a=sqrt(5)), written out so that the serial and the
    distributed FNO use one rule, and so that the scale is computed from the *global* ``fan_in``:
    a tensor-parallel shard only sees part of its input or output features, so a ``fan_in`` read from
    the local weight shape would be wrong.
    """
    with torch.no_grad():
        weight.normal_(0.0, 1.0 / math.sqrt(3.0 * fan_in))
        if bias is not None:
            bound = 1.0 / math.sqrt(fan_in)
            bias.uniform_(-bound, bound)


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

    ``init_k0`` (default: off) scales the initial weights by ``1/(1+(k/init_k0)^2)`` with the radial wavenumber ``k`` of
    each mode. Fields usually have little energy at high wavenumbers, so those weights get tiny gradients, hardly move
    from their random initial values, and then add noise: starting them small keeps that noise down.

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
        init_k0: Optional[float] = None,
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
        if init_k0 is not None:
            init = init * spectral_init_decay(spectral_wavenumbers(self.modes_kh, self.modes_kw), init_k0)
        self.weight = nn.Parameter(init)

    # The spectral layer runs eagerly even under torch.compile. Inductor lowers the FFTs to fallback kernels
    # that assert the output strides of the op's meta kernel, and with some torch versions cuFFT returns a
    # different layout (seen as `assert_size_stride ... expected stride 1==2049 at dim=2` for the c2c along h
    # at 4096^2). Keeping every FFT out of the compiled graph avoids the problem independently of the version;
    # the pointwise layers around it still compile, and they are what inductor fuses.
    @torch.compiler.disable
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


class FourierResample(nn.Module):
    """
    Band-limited resampling of a (B, C, H, W) field to ``out_size`` by truncating or zero-padding its spectrum:
    ``in_size -> out_size`` is a Fourier downsampling when the output is smaller and an upsampling when it is larger.
    The normalization preserves values (a constant stays that constant, and a field that is band-limited below the
    coarser grid's Nyquist frequency is sampled or interpolated exactly), and only the modes strictly below that
    Nyquist frequency are kept, see ``resample_modes``. The module has no parameters.

    This is how the FNO runs its blocks on a coarser latent grid: the activation memory of the blocks scales with
    the number of grid points they run on, which truncating the modes inside a spectral layer does not change.
    """

    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = tuple(in_size)
        self.out_size = tuple(out_size)
        self.keep_h, self.keep_w = resample_modes(min(self.in_size[0], self.out_size[0]), min(self.in_size[1], self.out_size[1]))

    # Eager under torch.compile, like the spectral layers, see FourierNeuralOperatorLayer.forward.
    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h2, w2 = self.out_size

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = torch.fft.rfft2(x.to(torch.float32), s=self.in_size, norm="forward")
            x = _truncate_modes(x.narrow(-1, 0, self.keep_w), self.keep_h, dim=-2)

            x = _pad_modes(x, self.keep_h, h2, dim=-2)
            x = F.pad(x, (0, w2 // 2 + 1 - self.keep_w))
            x = torch.fft.ifft(x, n=h2, dim=-2, norm="forward")
            x = _zero_self_conjugate_imag(x, w2, dim=-1)
            x = torch.fft.irfft(x, n=w2, dim=-1, norm="forward")

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

    def __init__(self, dim, height, width, modes_kh=None, modes_kw=None, mlp_ratio=2.0, act_layer=nn.GELU, init_k0=None):
        super().__init__()
        self.spectral = FourierNeuralOperatorLayer(dim, dim, height, width, modes_kh, modes_kw, init_k0=init_k0)
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
    Fourier neural operator: a pointwise lifting MLP, ``depth`` FNO blocks, and a pointwise projection MLP to
    ``out_chans``. Input and output are (B, C, H, W); all layers are channels-first, see distributed/fft.py.

    By default the blocks run on the full grid with ``embed_dim`` channels. With ``downsample = f > 1`` they run on a
    latent grid ``f`` times coarser per axis, which divides the activation memory of the blocks by ``f^2``:

        lift (full grid, ``lift_dim`` channels) -> Fourier downsample -> widen to ``embed_dim`` channels
        -> blocks on the latent grid -> narrow to ``lift_dim`` -> Fourier upsample (+ the lift, if
        ``resample_skip``) -> project (full grid)

    The few-channel full-grid parts are cheap, and the skip carries the wavenumbers above the latent grid's
    Nyquist frequency, which the latent path cannot represent. ``lift_dim`` defaults to ``embed_dim`` for
    ``downsample = 1`` (the layout of the original FNO, with the same parameters) and to 16 otherwise.
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
        spectral_init_k0=None,
        downsample=1,
        lift_dim=None,
        resample_skip=True,
        **kwargs,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        height, width = self.img_size
        if downsample < 1 or height % downsample or width % downsample:
            raise ValueError(f"downsample={downsample} must be a positive integer dividing the image size {self.img_size}")
        self.downsample = downsample
        self.latent_size = (height // downsample, width // downsample)
        lift_dim = lift_dim if lift_dim is not None else (embed_dim if downsample == 1 else 16)
        self.resample_skip = resample_skip and downsample > 1

        self.lift = MLP(in_chans, lift_dim, lift_dim)
        if downsample > 1:
            self.down = FourierResample(self.img_size, self.latent_size)
            self.up = FourierResample(self.latent_size, self.img_size)
        if lift_dim != embed_dim:
            self.expand = nn.Conv2d(lift_dim, embed_dim, 1)
            self.reduce = nn.Conv2d(embed_dim, lift_dim, 1)
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, *self.latent_size, modes_kh=modes_kh, modes_kw=modes_kw, mlp_ratio=mlp_ratio, init_k0=spectral_init_k0)
                for _ in range(depth)
            ]
        )
        self.project = MLP(lift_dim, lift_dim, out_chans)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):  # the 1x1 convs: lift, expand, reduce, skip, channel MLPs, project
                init_pointwise_(module.weight, module.bias, fan_in=module.in_channels)

    def forward(self, x):
        x = self.lift(x)
        skip = x if self.resample_skip else None
        if self.downsample > 1:
            x = self.down(x)
        if hasattr(self, "expand"):
            x = self.expand(x)
        for block in self.blocks:
            x = block(x)
        if hasattr(self, "reduce"):
            x = self.reduce(x)
        if self.downsample > 1:
            x = self.up(x)
        if skip is not None:
            x = x + skip
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
        spectral_init_k0=params.get("spectral_init_k0", None),
        downsample=params.get("downsample", 1),
        lift_dim=params.get("lift_dim", None),
        resample_skip=params.get("resample_skip", True),
        **kwargs,
    )
