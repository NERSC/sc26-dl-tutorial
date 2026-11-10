"""Spatially sharded Fourier resampling for the distributed FNO."""

import torch
from torch import nn

from utils import comm
from distributed.fft import DistributedRealFFT2, DistributedInverseRealFFT2
from networks.helpers import resample_modes


class DistributedFourierResample(nn.Module):
    """
    Distributed counterpart of ``networks.fno.FourierResample``: band-limited resampling of a spatially sharded
    channels-first field (B, C, H/sp1, W/sp2) to ``out_size``, returning the (B, C, H2/sp1, W2/sp2) shard of the
    result. It is a pencil FFT on the input grid followed by an inverse pencil FFT on the output grid, and the two
    already agree on the spectrum layout: both keep the same ``(kh, kw)`` modes, so each rank's spectral shard goes
    straight from one to the other. The normalization preserves values, like the serial operator.

    The channel count must be at least the size of each spatial group, like for every layer with a pencil FFT.
    """

    def __init__(self, in_size, out_size, h_comm: str = "sp1", w_comm: str = "sp2"):
        super().__init__()
        self.in_size = tuple(in_size)
        self.out_size = tuple(out_size)
        keep_h, keep_w = resample_modes(min(self.in_size[0], self.out_size[0]), min(self.in_size[1], self.out_size[1]))
        # explicit channels-first: (B, C, H, W)
        dims = dict(h_comm=h_comm, w_comm=w_comm, h_dim=2, w_dim=3, channel_dim=1)
        self.forward_transform = DistributedRealFFT2(*self.in_size, modes_kh=keep_h, modes_kw=keep_w, **dims)
        self.inverse_transform = DistributedInverseRealFFT2(*self.out_size, modes_kh=keep_h, modes_kw=keep_w, **dims)

    # Eager under torch.compile, like the spectral layers, see FourierNeuralOperatorLayer.forward.
    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.forward_transform(x.to(torch.float32), norm="forward")
            x = self.inverse_transform(x, norm="forward")
        return x.to(dtype)
