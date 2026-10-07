from dataclasses import dataclass
from typing import Sequence, Tuple

from distributed.helpers import compute_split_shapes


@dataclass(frozen=True)
class MeshSpatialShard:
    global_shape: Tuple[int, int]
    h_splits: Tuple[int, ...]
    w_splits: Tuple[int, ...]
    token_splits: Tuple[int, ...]
    h_rank: int
    w_rank: int

    @property
    def h_bounds(self) -> Tuple[int, int]:
        start = sum(self.h_splits[: self.h_rank])
        return start, start + self.h_splits[self.h_rank]

    @property
    def w_bounds(self) -> Tuple[int, int]:
        start = sum(self.w_splits[: self.w_rank])
        return start, start + self.w_splits[self.w_rank]

    @property
    def core_slices(self) -> Tuple[slice, slice]:
        h_start, h_end = self.h_bounds
        w_start, w_end = self.w_bounds
        return slice(h_start, h_end), slice(w_start, w_end)

    @property
    def local_shape(self) -> Tuple[int, int]:
        return self.h_splits[self.h_rank], self.w_splits[self.w_rank]


def make_vit_spatial_shard(
    global_shape: Sequence[int],
    patch_size: int,
    sp1_size: int,
    sp1_rank: int,
    sp2_size: int,
    sp2_rank: int,
    order: str,
) -> MeshSpatialShard:
    """Partition a ViT patch grid over its two spatial mesh axes."""
    height, width = map(int, global_shape)
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"ViT image shape {(height, width)} is not divisible by "
            f"patch size {patch_size}"
        )
    patch_row_splits = compute_split_shapes(height // patch_size, sp1_size)
    patch_col_splits = compute_split_shapes(width // patch_size, sp2_size)
    if min(patch_row_splits + patch_col_splits) <= 0:
        raise ValueError("spatial mesh creates an empty patch shard")

    spatial_order = [
        axis for axis in order.split("-") if axis in {"sp1", "sp2"}
    ]
    if set(spatial_order) != {"sp1", "sp2"}:
        raise ValueError(f"invalid spatial rank order {order!r}")
    axis_sizes = {"sp1": sp1_size, "sp2": sp2_size}
    token_splits = []
    for rank in range(sp1_size * sp2_size):
        coords = {}
        stride = 1
        for axis in spatial_order:
            coords[axis] = (rank // stride) % axis_sizes[axis]
            stride *= axis_sizes[axis]
        token_splits.append(
            patch_row_splits[coords["sp1"]]
            * patch_col_splits[coords["sp2"]]
        )

    return MeshSpatialShard(
        global_shape=(height, width),
        h_splits=tuple(rows * patch_size for rows in patch_row_splits),
        w_splits=tuple(cols * patch_size for cols in patch_col_splits),
        token_splits=tuple(token_splits),
        h_rank=sp1_rank,
        w_rank=sp2_rank,
    )


def make_grid_spatial_shard(
    global_shape: Sequence[int],
    sp1_size: int,
    sp1_rank: int,
    sp2_size: int,
    sp2_rank: int,
) -> MeshSpatialShard:
    """
    Partition a plain (H, W) grid over the two spatial mesh axes, for models without patches such
    as the FNO: sp1 splits H and sp2 splits W, balanced, with the same ``compute_split_shapes`` the
    FFT transposes use. ``token_splits`` is unused (there are no tokens) and left empty.
    """
    height, width = map(int, global_shape)
    h_splits = compute_split_shapes(height, sp1_size)
    w_splits = compute_split_shapes(width, sp2_size)
    if min(h_splits + w_splits) <= 0:
        raise ValueError("spatial mesh creates an empty shard")
    return MeshSpatialShard(
        global_shape=(height, width),
        h_splits=tuple(h_splits),
        w_splits=tuple(w_splits),
        token_splits=(),
        h_rank=sp1_rank,
        w_rank=sp2_rank,
    )
