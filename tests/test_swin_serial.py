from types import SimpleNamespace

import pytest
import torch

from networks.swin import (
    Swin,
    SwinTransformer,
    make_shifted_window_mask,
    window_partition,
    window_reverse,
)


def test_window_partition_round_trip():
    x = torch.randn(2, 8, 12, 5)
    windows = window_partition(x, (4, 3))
    assert windows.shape == (16, 12, 5)
    torch.testing.assert_close(window_reverse(windows, (4, 3), (8, 12)), x)


def test_shifted_window_mask_blocks_wrapped_regions():
    mask = make_shifted_window_mask((8, 8), (4, 4), (2, 2))
    assert mask.shape == (4, 16, 16)
    assert torch.all(torch.diagonal(mask, dim1=-2, dim2=-1) == 0)
    assert torch.any(mask == -100)


def test_swin_forward_backward_shape():
    model = SwinTransformer(
        img_size=(16, 16),
        patch_size=2,
        window_size=(4, 4),
        in_chans=3,
        out_chans=2,
        embed_dim=16,
        depth=2,
        num_heads=4,
        qkv_bias=True,
    )
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    y = model(x)
    assert y.shape == (2, 2, 16, 16)
    y.square().mean().backward()
    assert x.grad is not None


def test_swin_factory_uses_requested_window():
    params = SimpleNamespace(
        img_size=(16, 16),
        patch_size=2,
        window_size=(4, 4),
        n_in_channels=3,
        n_out_channels=2,
        embed_dim=16,
        depth=2,
        num_heads=4,
        dropout=0.0,
    )
    model = Swin(params)
    assert model.window_size == (4, 4)


def test_rejects_partial_windows():
    with pytest.raises(ValueError, match="divisible"):
        SwinTransformer(
            img_size=(12, 16),
            patch_size=2,
            window_size=(4, 4),
            embed_dim=16,
            depth=1,
            num_heads=4,
        )
