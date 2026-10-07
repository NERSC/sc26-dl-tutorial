import math
from typing import Optional

import torch
import torch.nn as nn

from utils import comm
from distributed.fft import DistributedRealFFT2, DistributedInverseRealFFT2
from networks.fno import init_pointwise_
from networks.vit_mp import MLP, ColumnParallelLinear, RowParallelLinear, _pointwise_linear


class FourierNeuralOperatorLayer(nn.Module):
    """
    Distributed Fourier neural operator layer, see networks/fno.py for the serial reference.

    The input is a spatially sharded channels-first field (B, in_channels, H/h_comm, W/w_comm)
    and the output is (B, out_channels, H/h_comm, W/w_comm), sharded the same way. The FNO runs
    channels-first, see distributed/fft.py for the reasoning.

    ``modes_kh`` and ``modes_kw`` set the number of retained wavenumbers after truncation
    (defaults: all). The weight has shape (in_channels, out_channels, kh_local, kw_local) and is
    sharded over the same groups as the spectrum, so no communication beyond the FFT transposes
    is needed. Note that the transposes swap the channel dim with h or w, so the channel count
    must be at least the size of each group.

    The inverse transform explicitly zeroes the imaginary parts of the self-conjugate w columns
    (DC and Nyquist) before the c2r transform, which is only defined for them being real.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        modes_kh: Optional[int] = None,
        modes_kw: Optional[int] = None,
        h_comm: str = "sp1",
        w_comm: str = "sp2",
        gain: float = 1.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # explicit channels-first: (B, C, H, W)
        dims = dict(h_dim=2, w_dim=3, channel_dim=1)
        self.forward_transform = DistributedRealFFT2(
            height, width, modes_kh=modes_kh, modes_kw=modes_kw, h_comm=h_comm, w_comm=w_comm, **dims
        )
        self.inverse_transform = DistributedInverseRealFFT2(
            height, width, modes_kh=modes_kh, modes_kw=modes_kw, h_comm=h_comm, w_comm=w_comm, **dims
        )

        # the pencil transposes swap the channel dim with h and with w, so channels are split too
        for group in (h_comm, w_comm):
            for name, channels in (("in_channels", in_channels), ("out_channels", out_channels)):
                if channels < comm.get_size(group):
                    raise ValueError(
                        f"{name}={channels} must be at least {group}={comm.get_size(group)}, "
                        "the transposes split the channels over that group"
                    )

        # wavenumbers after truncation and the local shard
        self.modes_kh = self.forward_transform.modes_kh
        self.modes_kw = self.forward_transform.modes_kw
        self.modes_kh_local = self.forward_transform.kh_shapes[comm.get_rank(h_comm)]
        self.modes_kw_local = self.forward_transform.kw_shapes[comm.get_rank(w_comm)]

        scale = math.sqrt(gain / in_channels)
        init = scale * torch.randn(
            in_channels,
            out_channels,
            self.modes_kh_local,
            self.modes_kw_local,
            dtype=torch.complex64,
        )
        self.weight = nn.Parameter(init)
        self.weight.comm_metadata = {
            "sharded": [None, None, h_comm, w_comm],
            "shared": ["tp"],
            "reduce": [],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.forward_transform(x.to(torch.float32))
            # channels are local after the forward transform, so this is a plain local contraction
            x = torch.einsum("bihw,iohw->bohw", x, self.weight)
            x = self.inverse_transform(x)

        return x.to(dtype)


class PointwiseLinear(nn.Module):
    """
    Channel-mixing 1x1 linear layer on a (B, C, H, W) field, replicated over all model-parallel
    ranks. As the field is spatially sharded, the weight gradient is a partial sum over sp1-sp2.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.comm_metadata = {
            "sharded": [None, None],
            "shared": ["tp-sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.bias.comm_metadata = {
                "sharded": [None],
                "shared": ["tp-sp1-sp2"],
                "reduce": ["sp1-sp2"],
            }
        else:
            self.register_parameter("bias", None)
        init_pointwise_(self.weight, self.bias, fan_in=in_features)

    def forward(self, x):
        return _pointwise_linear(x, self.weight, self.bias, "channels_first")


class Block(nn.Module):
    """
    Distributed FNO block, ``x = act(spectral(x) + skip(x))`` followed by an optional residual
    channel MLP, on spatially sharded (B, C, H/sp1, W/sp2) fields. The MLP is the ViT's, in its
    channels-first mode, and is tensor-parallel over tp.
    """

    def __init__(self, dim, height, width, modes_kh=None, modes_kw=None, mlp_ratio=2.0, act_layer=nn.GELU):
        super().__init__()
        self.spectral = FourierNeuralOperatorLayer(dim, dim, height, width, modes_kh, modes_kw)
        self.skip = PointwiseLinear(dim, dim)
        self.act = act_layer()
        self.mlp = (
            MLP(dim, int(dim * mlp_ratio), dim, act_layer=act_layer, layout="channels_first")
            if mlp_ratio > 0
            else None
        )

    def forward(self, x):
        x = self.act(self.spectral(x) + self.skip(x))
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class FourierNeuralOperator(nn.Module):
    """
    Distributed Fourier neural operator, see networks/fno.py for the serial reference.

    Spatially sharded at its interface, like the ViT: every model-parallel rank takes its own
    (B, in_chans, H/sp1, W/sp2) shard as input, as loaded by the data loader from the same
    ``distributed.spatial_sharding.make_grid_spatial_shard`` shard, and returns its own
    (B, out_chans, H/sp1, W/sp2) shard. Nothing is scattered or gathered, so the loss and the
    metrics reduce over the spatial groups (``utils.loss.mse_loss``). The pointwise layers are
    tensor-parallel over tp; the spectral layers are sharded over the wavenumbers.
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
        height, width = self.img_size

        self.lift = MLP(in_chans, embed_dim, embed_dim, layout="channels_first")
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, height, width, modes_kh=modes_kh, modes_kw=modes_kw, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.project = MLP(embed_dim, embed_dim, out_chans, layout="channels_first")

        # The tensor-parallel layers are the ViT's, which initialize with a constant std of 0.02. That is
        # right at the ViT's width of ~1024 but vanishes the signal at the FNO's width, so use the same
        # fan-in scaled rule as the serial FNO, with the *global* fan-in: a column-parallel weight is
        # sharded over its outputs (the local input width is the global one), a row-parallel one over its
        # inputs. Replicated parameters (the row-parallel bias, the sp-replicated layers) are drawn
        # independently on every rank here and synchronized from the group root by
        # init_params_for_shared_weights, as for every other replicated parameter.
        tp = comm.get_size("tp")
        for module in self.modules():
            if isinstance(module, ColumnParallelLinear):
                init_pointwise_(module.weight, module.bias, fan_in=module.weight.shape[1])
            elif isinstance(module, RowParallelLinear):
                init_pointwise_(module.weight, module.bias, fan_in=module.in_features_local * tp)

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
