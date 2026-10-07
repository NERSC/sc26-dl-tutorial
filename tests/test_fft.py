import unittest

import torch
import torch.distributed as dist
from parameterized import parameterized

from tests.helpers import DistributedTestCase
from distributed.fft import DistributedRealFFT2, DistributedInverseRealFFT2

# (h_dim, w_dim, channel_dim): channels-first is the default and what the FNO uses, channels-last
# checks that the transforms are layout-general
LAYOUTS = {"last": (1, 2, 3), "first": (2, 3, 1)}


def to_layout(x, dims):
    """permute a canonical (B, C, H, W) tensor into the layout given by (h_dim, w_dim, channel_dim)"""
    h, w, c = dims
    perm = [0] * 4
    perm[c], perm[h], perm[w] = 1, 2, 3
    return x.permute(*perm).contiguous()


def from_layout(x, dims):
    """inverse of to_layout"""
    h, w, c = dims
    return x.permute(0, c, h, w).contiguous()


class TestDistributedFFT(DistributedTestCase):
    @parameterized.expand(
        [
            [layout, *cfg]
            for layout in LAYOUTS
            for cfg in [
                [64, 128, 4, 6, None, None],
                [61, 122, 3, 5, None, None],
                [64, 128, 4, 6, 24, 20],
            ]
        ]
    )
    def test_fft2_forward_inverse(self, layout, height, width, batch, chans, modes_kh, modes_kw):
        dims = LAYOUTS[layout]
        h, w, c = dims
        fwd = DistributedRealFFT2(height, width, modes_kh, modes_kw, h_dim=h, w_dim=w, channel_dim=c).to(self.device)
        inv = DistributedInverseRealFFT2(height, width, modes_kh, modes_kw, h_dim=h, w_dim=w, channel_dim=c).to(self.device)
        truncated = fwd.modes_kh < height or fwd.modes_kw < width // 2 + 1

        # reference in canonical (B, C, H, W), identical on all ranks
        full = torch.randn(batch, chans, height, width, device=self.device)
        dist.broadcast(full, src=0)
        full.requires_grad_(True)
        ref = torch.fft.rfft2(full, norm="ortho")[..., : fwd.modes_kw]
        if fwd.modes_kh < height:
            high, low = fwd.modes_kh - fwd.modes_kh // 2, fwd.modes_kh // 2
            ref = torch.cat([ref[..., :high, :], ref[..., height - low :, :]], dim=-2)

        # distributed forward
        x = self._split(to_layout(full.detach(), dims), h, w).clone().requires_grad_(True)
        y = fwd(x)
        y_full = from_layout(self._gather(y.detach(), fwd.modes_kh, fwd.modes_kw, h, w), dims)
        self.assertTrue(torch.allclose(y_full, ref, atol=1e-4, rtol=1e-4))

        # gradient through the all-to-alls matches the single-process gradient
        (y.abs() ** 2).sum().backward()
        (ref.abs() ** 2).sum().backward()
        gx = self._split(to_layout(full.grad, dims), h, w)
        self.assertTrue(torch.allclose(x.grad, gx, atol=1e-3, rtol=1e-3))

        # round trip
        if not truncated:
            z = inv(y.detach())
            z_full = from_layout(self._gather(z, height, width, h, w), dims)
            self.assertTrue(torch.allclose(z_full, full.detach(), atol=1e-4, rtol=1e-4))

    def test_default_layout_is_channels_first(self):
        fwd = DistributedRealFFT2(32, 64)
        self.assertEqual((fwd.h_dim, fwd.w_dim, fwd.channel_dim), LAYOUTS["first"])

    def test_dims_validation(self):
        with self.assertRaises(ValueError):
            DistributedRealFFT2(32, 64, h_dim=1, w_dim=1, channel_dim=3)


if __name__ == "__main__":
    unittest.main()
