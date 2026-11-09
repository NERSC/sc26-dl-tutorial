import torch
import torch.distributed as dist
from parameterized import parameterized

from tests.helpers import DistributedTestCase
from distributed.resample import DistributedFourierResample
from networks.fno import FourierResample


class TestDistributedFourierResample(DistributedTestCase):
    """The distributed resampling of the shards equals the serial resampling of the full fields, in both directions,
    on uneven grids, with the gradients."""

    @parameterized.expand(
        [
            [(64, 96), (32, 48)],  # down by 2
            [(48, 48), (16, 16)],  # down by 3
            [(66, 90), (33, 45)],  # odd coarse grid: uneven shards over the groups
            [(32, 48), (64, 96)],  # up by 2
            [(33, 45), (66, 90)],
        ]
    )
    def test_matches_serial(self, in_size, out_size):
        c = 8
        full = torch.randn(2, c, *in_size, device=self.device)
        dist.broadcast(full, src=0)
        x_ref = full.clone().requires_grad_(True)
        y_ref = FourierResample(in_size, out_size).to(self.device)(x_ref)

        x = self._split(full).clone().requires_grad_(True)
        y = DistributedFourierResample(in_size, out_size).to(self.device)(x)
        torch.testing.assert_close(y, self._split(y_ref), atol=1e-5, rtol=1e-4)

        # gradient through the all-to-alls, from the loss of the local shard
        y_ref.square().sum().backward()
        y.square().sum().backward()
        torch.testing.assert_close(x.grad, self._split(x_ref.grad), atol=1e-4, rtol=1e-3)

    def test_round_trip_of_a_band_limited_field(self):
        """down then up through the distributed operators returns a field below the coarse Nyquist frequency"""
        big, small = (64, 96), (32, 48)
        low = FourierResample(big, small).to(self.device)
        z = torch.randn(2, 8, *small, device=self.device)
        dist.broadcast(z, src=0)
        x = FourierResample(small, big).to(self.device)(z)  # band-limited on the big grid
        y = DistributedFourierResample(small, big).to(self.device)(self._split(z))
        torch.testing.assert_close(y, self._split(x), atol=1e-5, rtol=1e-4)
        back = DistributedFourierResample(big, small).to(self.device)(y)
        torch.testing.assert_close(back, self._split(low(x)), atol=1e-5, rtol=1e-4)

    def test_channels_below_the_group_size_fail_early(self):
        from utils import comm

        sp = max(comm.get_size("sp1"), comm.get_size("sp2"))
        if sp > 1:
            from networks.fno_mp import FourierNeuralOperator as D

            with self.assertRaises(ValueError):
                D((32, 64), 4, 4, 16, 1, 4, 4, downsample=2, lift_dim=sp - 1)
