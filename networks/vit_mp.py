from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.helpers import DropPath, trunc_normal_
from utils import comm
from distributed.mappings import (
    allgather_forward_reducescatter_backward,
    allreduce_forward_identity_backward,
    identity_forward_allreduce_backward,
)


_LAYOUTS = ("channels_last", "channels_first")


def _check_layout(layout):
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {_LAYOUTS}, got {layout!r}")
    return layout


def _pointwise_linear(x, weight, bias, layout):
    """
    Linear map over the channel dim, for either layout. The weight is (out, in) in both, so
    parameters, sharding and checkpoints do not depend on the layout.

    channels_last:  x is (..., C), F.linear.
    channels_first: x is (B, C, *spatial), a 1x1 conv1d over the flattened spatial dims
                    (a free view for contiguous inputs), so no permute copies are needed.
    """
    if layout == "channels_last":
        return F.linear(x, weight, bias)
    spatial = x.shape[2:]
    y = F.conv1d(x.flatten(2), weight.unsqueeze(-1), bias)
    return y.reshape(y.shape[0], y.shape[1], *spatial)


def _add_channel_bias(x, bias, layout):
    if bias is None:
        return x
    if layout == "channels_last":
        return x + bias
    # Channels-first is the FNO's layout: add the bias in the activation dtype. Under autocast the conv output is
    # bf16 and the bias an fp32 parameter, so `x + bias` would promote the whole residual stream (and what is saved
    # downstream of it) to fp32, doubling the memory of every later tensor and differing from the serial model, whose
    # convs apply the bias through autocast. (The ViT's channels-last path is left as it is.)
    return x + bias.to(x.dtype).view(1, -1, *([1] * (x.dim() - 2)))


class ColumnParallelLinear(nn.Module):
    """Linear layer sharded over its stored output-feature dimension."""

    def __init__(self, in_features, out_features, bias=True, layout="channels_last"):
        super().__init__()
        self.layout = _check_layout(layout)
        tp_size = comm.get_size("tp")
        if out_features % tp_size != 0:
            raise ValueError(
                f"out_features={out_features} is not divisible by tp={tp_size}"
            )
        self.out_features_local = out_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(self.out_features_local, in_features)
        )
        self.weight.comm_metadata = {
            "sharded": ["tp", None],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_local))
            self.bias.comm_metadata = {
                "sharded": ["tp"],
                "shared": ["sp1-sp2"],
                "reduce": ["sp1-sp2"],
            }
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        x = identity_forward_allreduce_backward(x, "tp")
        return _pointwise_linear(x, self.weight, self.bias, self.layout)


class RowParallelLinear(nn.Module):
    """Linear layer sharded over its stored input-feature dimension."""

    def __init__(self, in_features, out_features, bias=True, layout="channels_last"):
        super().__init__()
        self.layout = _check_layout(layout)
        tp_size = comm.get_size("tp")
        if in_features % tp_size != 0:
            raise ValueError(
                f"in_features={in_features} is not divisible by tp={tp_size}"
            )
        self.in_features_local = in_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_local)
        )
        self.weight.comm_metadata = {
            "sharded": [None, "tp"],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.bias.comm_metadata = {
                "sharded": [None],
                "shared": ["tp-sp1-sp2"],
                "reduce": ["sp1-sp2"],
            }
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        x = _pointwise_linear(x, self.weight, None, self.layout)
        x = allreduce_forward_identity_backward(x, "tp")
        return _add_channel_bias(x, self.bias, self.layout)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        layout="channels_last",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # layout: "channels_last" (..., C), as used by the ViT, or "channels_first" (B, C, *spatial)
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, layout=layout)
        self.act = act_layer()
        self.fc2 = RowParallelLinear(hidden_features, out_features, layout=layout)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        spatial_shapes=None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        tp_size = comm.get_size("tp")
        if num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads={num_heads} is not divisible by tp={tp_size}"
            )
        self.num_heads = num_heads
        self.num_heads_local = num_heads // tp_size
        self.head_dim = dim // num_heads
        self.spatial_shapes = spatial_shapes
        self.q = ColumnParallelLinear(dim, dim, bias=qkv_bias)
        self.k = ColumnParallelLinear(dim, dim, bias=qkv_bias)
        self.v = ColumnParallelLinear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = RowParallelLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, sequence_local, _ = x.shape
        q = self.q(x).reshape(
            batch, sequence_local, self.num_heads_local, self.head_dim
        ).permute(0, 2, 1, 3)
        k = self.k(x).reshape(
            batch, sequence_local, self.num_heads_local, self.head_dim
        ).permute(0, 2, 1, 3)
        v = self.v(x).reshape(
            batch, sequence_local, self.num_heads_local, self.head_dim
        ).permute(0, 2, 1, 3)

        k = allgather_forward_reducescatter_backward(
            k, dim=2, shapes=self.spatial_shapes, comm_name="sp1-sp2"
        )
        v = allgather_forward_reducescatter_backward(
            v, dim=2, shapes=self.spatial_shapes, comm_name="sp1-sp2"
        )
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(
            batch,
            sequence_local,
            self.num_heads_local * self.head_dim,
        )
        return self.proj_drop(self.proj(x))


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, **kwargs):
        super().__init__(normalized_shape, **kwargs)
        if self.elementwise_affine:
            self.weight.comm_metadata = {
                "sharded": [None],
                "shared": ["tp-sp1-sp2"],
                "reduce": ["sp1-sp2"],
            }
            if self.bias is not None:
                self.bias.comm_metadata = {
                    "sharded": [None],
                    "shared": ["tp-sp1-sp2"],
                    "reduce": ["sp1-sp2"],
                }


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        spatial_shapes=None,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            spatial_shapes=spatial_shapes,
        )
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = self.h * self.w
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.proj.weight.comm_metadata = {
            "sharded": [None, None, None, None],
            "shared": ["tp-sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.proj.bias.comm_metadata = {
            "sharded": [None],
            "shared": ["tp-sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        in_chans=3,
        out_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=LayerNorm,
        *,
        spatial_shard,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = tuple(img_size)
        self.spatial_shard = spatial_shard
        self.out_ch = out_chans
        if spatial_shard.global_shape != self.img_size:
            raise ValueError(
                "spatial shard global shape does not match ViT image size"
            )
        local_img_size = spatial_shard.local_shape
        self.patch_embed = PatchEmbed(
            img_size=local_img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.spatial_shapes = list(spatial_shard.token_splits)
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.patch_embed.h,
                self.patch_embed.w,
                embed_dim,
            )
        )
        self.pos_embed.comm_metadata = {
            "sharded": [None, "sp1", "sp2", None],
            "shared": ["tp"],
            "reduce": [],
        }
        self.pos_drop = nn.Dropout(p=drop_rate)
        drop_path = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path[index],
                    norm_layer=norm_layer,
                    spatial_shapes=self.spatial_shapes,
                )
                for index in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.out_size = out_chans * patch_size * patch_size
        self.head = nn.Linear(embed_dim, self.out_size, bias=False)
        self.head.weight.comm_metadata = {
            "sharded": [None, None],
            "shared": ["tp-sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def prepare_tokens(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        return x.flatten(1, 2)

    def forward_head(self, x):
        batch = x.shape[0]
        x = x.reshape(
            batch,
            self.patch_embed.h,
            self.patch_embed.w,
            self.embed_dim,
        )
        x = self.head(x)
        x = x.reshape(
            batch,
            self.patch_embed.h,
            self.patch_embed.w,
            self.patch_size,
            self.patch_size,
            self.out_ch,
        )
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(
            batch,
            self.out_ch,
            self.patch_embed.img_size[0],
            self.patch_embed.img_size[1],
        )

    def forward(self, x):
        x = self.prepare_tokens(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.forward_head(x)


def ViT(params, spatial_shard, **kwargs):
    return VisionTransformer(
        img_size=tuple(params.img_size),
        in_chans=params.n_in_channels,
        out_chans=params.n_out_channels,
        patch_size=params.patch_size,
        embed_dim=params.embed_dim,
        depth=params.depth,
        num_heads=params.num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6),
        drop_path_rate=float(params.dropout),
        drop_rate=float(params.dropout),
        attn_drop_rate=float(params.dropout),
        spatial_shard=spatial_shard,
        **kwargs,
    )
