from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from distributed.mappings import roll_forward_reverseroll_backward
from networks.helpers import DropPath, trunc_normal_
from networks.swin import window_partition, window_reverse
from networks.vit_mp import (
    ColumnParallelLinear,
    LayerNorm,
    MLP,
    RowParallelLinear,
)
from utils import comm


def make_local_shifted_window_mask(
    global_feature_size,
    local_patch_slices,
    window_size,
    shift_size,
):
    """Build a standard global Swin mask and retain this rank's windows."""
    if not any(shift_size):
        return None
    height, width = global_feature_size
    window_h, window_w = window_size
    shift_h, shift_w = shift_size
    mask = torch.zeros((1, height, width, 1))
    h_slices = (
        slice(0, -window_h),
        slice(-window_h, -shift_h),
        slice(-shift_h, None),
    )
    w_slices = (
        slice(0, -window_w),
        slice(-window_w, -shift_w),
        slice(-shift_w, None),
    )
    region = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            mask[:, h_slice, w_slice, :] = region
            region += 1
    local_h, local_w = local_patch_slices
    mask = window_partition(mask[:, local_h, local_w, :], window_size)
    mask = mask.squeeze(-1)
    mask = mask.unsqueeze(1) - mask.unsqueeze(2)
    return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        tp_size = comm.get_size("tp")
        if num_heads % tp_size:
            raise ValueError(
                f"num_heads={num_heads} is not divisible by tp={tp_size}"
            )
        self.num_heads_local = num_heads // tp_size
        self.head_dim = dim // num_heads
        self.q = ColumnParallelLinear(dim, dim, bias=qkv_bias)
        self.k = ColumnParallelLinear(dim, dim, bias=qkv_bias)
        self.v = ColumnParallelLinear(dim, dim, bias=qkv_bias)
        self.attn_drop = float(attn_drop)
        self.proj = RowParallelLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        batch_windows, tokens, _ = x.shape

        def project(layer):
            return (
                layer(x)
                .reshape(
                    batch_windows,
                    tokens,
                    self.num_heads_local,
                    self.head_dim,
                )
                .permute(0, 2, 1, 3)
            )

        q, k, v = project(self.q), project(self.k), project(self.v)
        if mask is not None:
            num_windows = mask.shape[0]
            mask = (
                mask.to(device=q.device, dtype=q.dtype)
                .unsqueeze(0)
                .expand(
                    batch_windows // num_windows,
                    num_windows,
                    tokens,
                    tokens,
                )
                .reshape(batch_windows, 1, tokens, tokens)
            )
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(
            batch_windows,
            tokens,
            self.num_heads_local * self.head_dim,
        )
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        feature_size,
        global_feature_size,
        local_patch_slices,
        window_size,
        shift_size=(0, 0),
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=LayerNorm,
    ):
        super().__init__()
        self.feature_size = tuple(feature_size)
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)
        for feature, window, shift in zip(
            self.feature_size, self.window_size, self.shift_size
        ):
            if feature % window:
                raise ValueError(
                    f"local feature size {self.feature_size} must be divisible "
                    f"by window size {self.window_size}"
                )
            if not 0 <= shift < window:
                raise ValueError("shift must be smaller than its window")

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )
        self.register_buffer(
            "attn_mask",
            make_local_shifted_window_mask(
                global_feature_size,
                local_patch_slices,
                self.window_size,
                self.shift_size,
            ),
            persistent=False,
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if any(self.shift_size):
            x = roll_forward_reverseroll_backward(
                x,
                shifts=tuple(-shift for shift in self.shift_size),
                dims=(1, 2),
                comm_names=("sp1", "sp2"),
            )
        x = window_partition(x, self.window_size)
        x = self.attn(x, self.attn_mask)
        x = window_reverse(x, self.window_size, self.feature_size)
        if any(self.shift_size):
            x = roll_forward_reverseroll_backward(
                x,
                shifts=self.shift_size,
                dims=(1, 2),
                comm_names=("sp1", "sp2"),
            )
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        if img_size[0] % patch_size or img_size[1] % patch_size:
            raise ValueError("local image dimensions must divide patch_size")
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size
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


class DistributedSwinTransformer(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        window_size=(8, 8),
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
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.window_size = tuple(window_size)
        self.embed_dim = embed_dim
        self.out_ch = out_chans
        self.spatial_shard = spatial_shard
        if spatial_shard.global_shape != self.img_size:
            raise ValueError(
                "spatial shard global shape does not match Swin image size"
            )

        self.patch_embed = PatchEmbed(
            img_size=spatial_shard.local_shape,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        local_feature_size = (self.patch_embed.h, self.patch_embed.w)
        global_feature_size = (
            self.img_size[0] // patch_size,
            self.img_size[1] // patch_size,
        )
        h_slice, w_slice = spatial_shard.core_slices
        local_patch_slices = (
            slice(h_slice.start // patch_size, h_slice.stop // patch_size),
            slice(w_slice.start // patch_size, w_slice.stop // patch_size),
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                local_feature_size[0],
                local_feature_size[1],
                embed_dim,
            )
        )
        self.pos_embed.comm_metadata = {
            "sharded": [None, "sp1", "sp2", None],
            "shared": ["tp"],
            "reduce": [],
        }
        self.pos_drop = nn.Dropout(drop_rate)
        drop_paths = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    feature_size=local_feature_size,
                    global_feature_size=global_feature_size,
                    local_patch_slices=local_patch_slices,
                    window_size=self.window_size,
                    shift_size=(
                        (0, 0)
                        if index % 2 == 0
                        else (
                            self.window_size[0] // 2,
                            self.window_size[1] // 2,
                        )
                    ),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_paths[index],
                    norm_layer=norm_layer,
                )
                for index in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim,
            out_chans * patch_size * patch_size,
            bias=False,
        )
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

    def forward_head(self, x):
        batch, height, width, _ = x.shape
        x = self.head(x).reshape(
            batch,
            height,
            width,
            self.patch_size,
            self.patch_size,
            self.out_ch,
        )
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(
            batch,
            self.out_ch,
            height * self.patch_size,
            width * self.patch_size,
        )

    def forward(self, x):
        x = self.pos_drop(self.patch_embed(x) + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.forward_head(self.norm(x))


def Swin(params, spatial_shard, **kwargs):
    return DistributedSwinTransformer(
        img_size=tuple(params.img_size),
        in_chans=params.n_in_channels,
        out_chans=params.n_out_channels,
        patch_size=params.patch_size,
        window_size=tuple(params.window_size),
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
