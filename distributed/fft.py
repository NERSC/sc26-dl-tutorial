from typing import Optional

import torch
from torch import nn

from utils import comm
from distributed.helpers import compute_split_shapes
from distributed.mappings import distributed_transpose
from networks.helpers import _truncate_modes, _pad_modes, _zero_self_conjugate_imag


# Tensor layout: the transforms below take the position of the h, w and channel dims explicitly.
# The default is channels-first, (B, C, H, W), which is what the FNO uses: the FFT runs along the
# two trailing, contiguous dims, which is 1.5-2x faster than along strided dims for channels-last
# (measured with cuFFT on a single GPU), and the pointwise layers lose nothing, since the shared
# MLP layers in networks/vit_mp.py take layout="channels_first" and run as a 1x1 conv1d on the
# flattened spatial dims instead of F.linear, so no permute copies are needed. The ViT keeps its
# channels-last (B, L, E) tokens and the MLP's default layout. Any other layout works here by passing
# the dims, e.g. (B, H, W, C) is h_dim=1, w_dim=2, channel_dim=3.


def _contiguous_complex_safe(x):
    """contiguous() through the real view for complex tensors (inductor cannot copy complex buffers)."""
    if x.is_complex():
        return torch.view_as_complex(torch.view_as_real(x).contiguous())
    return x.contiguous()


class _DistributedFFT2Base(nn.Module):
    """Shared setup of the forward and inverse transforms."""

    def __init__(
        self,
        height: int,
        width: int,
        modes_kh: Optional[int] = None,
        modes_kw: Optional[int] = None,
        h_comm: str = "sp1",
        w_comm: str = "sp2",
        h_dim: int = 2,
        w_dim: int = 3,
        channel_dim: int = 1,
    ):
        super().__init__()

        self.h_comm = h_comm
        self.w_comm = w_comm
        self.comm_size_h = comm.get_size(h_comm)
        self.comm_size_w = comm.get_size(w_comm)

        self.h_dim = h_dim
        self.w_dim = w_dim
        self.channel_dim = channel_dim
        if len({h_dim % 4, w_dim % 4, channel_dim % 4}) != 3:
            raise ValueError(
                f"h_dim, w_dim and channel_dim must be distinct dims of a 4D tensor, "
                f"got {h_dim}, {w_dim}, {channel_dim}"
            )

        self.height = height
        self.width = width
        self.modes_kh = min(modes_kh or self.height, self.height)
        self.modes_kw = min(modes_kw or self.width // 2 + 1, self.width // 2 + 1)

        # every group rank needs a non-empty shard of the spatial dims and of the retained modes
        for name, size, group, group_size in [
            ("height", self.height, h_comm, self.comm_size_h),
            ("modes_kh", self.modes_kh, h_comm, self.comm_size_h),
            ("width", self.width, w_comm, self.comm_size_w),
            ("modes_kw", self.modes_kw, w_comm, self.comm_size_w),
        ]:
            if size < group_size:
                raise ValueError(f"{name}={size} must be at least {group}={group_size}, it is split over that group")

        self.h_shapes = compute_split_shapes(self.height, self.comm_size_h)
        self.w_shapes = compute_split_shapes(self.width, self.comm_size_w)
        self.kh_shapes = compute_split_shapes(self.modes_kh, self.comm_size_h)
        self.kw_shapes = compute_split_shapes(self.modes_kw, self.comm_size_w)


class DistributedRealFFT2(_DistributedFFT2Base):
    """
    2D real FFT of a 4D tensor with H split over `h_comm` and W split over `w_comm`
    (pencil decomposition). The input is spatially sharded, the output is spectrally
    sharded: the h and w dims then hold modes_kh/h_comm and modes_kw/w_comm modes, and the channel
    dim is local and full. Default layout is (B, C, H, W), see above.
    """

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho") -> torch.Tensor:
        h, w, c = self.h_dim, self.w_dim, self.channel_dim
        num_chans = x.shape[c]

        # make w local by transposing into the channel dim
        if self.comm_size_w > 1:
            x = distributed_transpose(x, (c, w), self.w_shapes, self.w_comm)

        x = torch.fft.rfft(x, n=self.width, dim=w, norm=norm)
        x = _contiguous_complex_safe(x.narrow(w, 0, self.modes_kw))

        # kw is split, c is local
        if self.comm_size_w > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_w)
            x = distributed_transpose(x, (w, c), chan_shapes, self.w_comm)

        # make h local by transposing into the channel dim
        if self.comm_size_h > 1:
            x = distributed_transpose(x, (c, h), self.h_shapes, self.h_comm)

        x = torch.fft.fft(x, n=self.height, dim=h, norm=norm)
        x = _truncate_modes(x, self.modes_kh, dim=h)

        # kh is split, c is local
        if self.comm_size_h > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_h)
            x = distributed_transpose(x, (h, c), chan_shapes, self.h_comm)

        return x


class DistributedInverseRealFFT2(_DistributedFFT2Base):
    """Inverse of DistributedRealFFT2: spectrally sharded in, spatially sharded out."""

    def forward(self, x: torch.Tensor, norm: Optional[str] = "ortho") -> torch.Tensor:
        h, w, c = self.h_dim, self.w_dim, self.channel_dim
        num_chans = x.shape[c]

        # channels are split, kh is local
        if self.comm_size_h > 1:
            x = distributed_transpose(x, (c, h), self.kh_shapes, self.h_comm)

        # zero-fill the truncated middle modes
        x = _pad_modes(x, self.modes_kh, self.height, dim=h)
        x = torch.fft.ifft(x, n=self.height, dim=h, norm=norm)

        # kh is split again, c is local
        if self.comm_size_h > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_h)
            x = distributed_transpose(x, (h, c), chan_shapes, self.h_comm)

        # channels are split, kw is local
        if self.comm_size_w > 1:
            x = distributed_transpose(x, (c, w), self.kw_shapes, self.w_comm)

        # c2r is only defined for real DC / Nyquist columns, zero their imaginary parts
        x = _zero_self_conjugate_imag(x, self.width, dim=w)
        x = torch.fft.irfft(x, n=self.width, dim=w, norm=norm)

        # w is split, c is local
        if self.comm_size_w > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_w)
            x = distributed_transpose(x, (w, c), chan_shapes, self.w_comm)

        return x
