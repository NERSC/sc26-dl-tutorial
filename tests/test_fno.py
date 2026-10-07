import torch
import torch.distributed as dist
from parameterized import parameterized

from utils import comm
from tests.helpers import DistributedTestCase
from distributed.helpers import init_params_for_shared_weights
from networks.fno import FourierNeuralOperatorLayer
from networks.fno_mp import FourierNeuralOperatorLayer as DistributedFNOLayer


class TestDistributedFNO(DistributedTestCase):
    def _copy_weight(self, layer, layer_dist):
        """copy the (kh, kw) shard owned by this rank into the distributed layer"""
        h0 = sum(layer_dist.forward_transform.kh_shapes[: comm.get_rank("sp1")])
        w0 = sum(layer_dist.forward_transform.kw_shapes[: comm.get_rank("sp2")])
        with torch.no_grad():
            layer_dist.weight.copy_(
                layer.weight[
                    :, :, h0 : h0 + layer_dist.modes_kh_local, w0 : w0 + layer_dist.modes_kw_local
                ]
            )

    @parameterized.expand(
        [
            [32, 64, 4, 6, None, None],
            [31, 62, 6, 5, None, None],
            [32, 64, 4, 6, 12, 9],
            [33, 66, 5, 9, 11, 17],
        ]
    )
    def test_matches_serial(self, height, width, cin, cout, kh, kw):
        layer = FourierNeuralOperatorLayer(cin, cout, height, width, kh, kw).to(self.device)
        layer_dist = DistributedFNOLayer(cin, cout, height, width, kh, kw).to(self.device)
        self.assertEqual(layer.modes_kh, layer_dist.modes_kh)
        self.assertEqual(layer.modes_kw, layer_dist.modes_kw)

        # same weights and input on all ranks
        with torch.no_grad():
            for p in layer.parameters():
                dist.broadcast(p, src=0)
        self._copy_weight(layer, layer_dist)
        full = torch.randn(2, cin, height, width, device=self.device)
        dist.broadcast(full, src=0)

        x = full.clone().requires_grad_(True)
        x_dist = self._split(full).clone().requires_grad_(True)

        y = layer(x)
        y_dist = layer_dist(x_dist)
        self.assertEqual(y_dist.shape[1], cout)
        torch.testing.assert_close(
            self._gather(y_dist.detach(), height, width), y.detach(), atol=1e-4, rtol=1e-4
        )

        # input and (sharded) weight gradients
        y.square().sum().backward()
        y_dist.square().sum().backward()
        torch.testing.assert_close(x_dist.grad, self._split(x.grad), atol=1e-3, rtol=1e-3)
        h0 = sum(layer_dist.forward_transform.kh_shapes[: comm.get_rank("sp1")])
        w0 = sum(layer_dist.forward_transform.kw_shapes[: comm.get_rank("sp2")])
        expected = layer.weight.grad[
            :, :, h0 : h0 + layer_dist.modes_kh_local, w0 : w0 + layer_dist.modes_kw_local
        ]
        torch.testing.assert_close(layer_dist.weight.grad, expected, atol=1e-3, rtol=1e-3)

    def test_group_size_validation(self):
        """channel counts, spatial sizes and mode counts below a group size fail at construction"""
        sp1, sp2 = comm.get_size("sp1"), comm.get_size("sp2")
        if sp1 > 1:
            for kwargs in (dict(in_channels=sp1 - 1, out_channels=4), dict(in_channels=4, out_channels=sp1 - 1), dict(in_channels=4, out_channels=4, modes_kh=sp1 - 1)):
                with self.assertRaises(ValueError):
                    DistributedFNOLayer(height=32, width=64, **kwargs)
        if sp2 > 1:
            with self.assertRaises(ValueError):
                DistributedFNOLayer(4, 4, 32, 64, modes_kw=sp2 - 1)

    def test_comm_metadata(self):
        layer = DistributedFNOLayer(4, 4, 32, 64)
        init_params_for_shared_weights(layer)  # validates the metadata
