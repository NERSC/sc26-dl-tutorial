import torch
from parameterized import parameterized

from utils import comm
from tests.helpers import DistributedTestCase
from networks.vit import MLP as SerialMLP
from networks.vit_mp import MLP, ColumnParallelLinear, RowParallelLinear


def copy_serial_weights(serial, dist_mlp):
    """copy the serial MLP's weights into this rank's tp shard of the distributed one"""
    tp, rank = comm.get_size("tp"), comm.get_rank("tp")
    n = serial.fc1.weight.shape[0] // tp
    s, e = rank * n, (rank + 1) * n
    with torch.no_grad():
        dist_mlp.fc1.weight.copy_(serial.fc1.weight[s:e])
        dist_mlp.fc1.bias.copy_(serial.fc1.bias[s:e])
        dist_mlp.fc2.weight.copy_(serial.fc2.weight[:, s:e])
        dist_mlp.fc2.bias.copy_(serial.fc2.bias)


class TestMLPLayout(DistributedTestCase):
    """Both layouts of the shared MLP against the serial channels-last reference, under tp"""

    @parameterized.expand([["channels_last"], ["channels_first"]])
    def test_mlp_matches_serial(self, layout):
        c, hid = 8, 32
        serial = SerialMLP(c, hid).to(self.device)
        mlp = MLP(c, hid, layout=layout).to(self.device)
        copy_serial_weights(serial, mlp)

        # (B, H, W, C) for the serial reference, the same field in the layout under test
        x = torch.randn(2, 5, 7, c, device=self.device)
        torch.distributed.broadcast(x, src=0)
        x_ref = x.clone().requires_grad_(True)
        x_in = (x.permute(0, 3, 1, 2) if layout == "channels_first" else x).contiguous().requires_grad_(True)

        y_ref = serial(x_ref)
        y = mlp(x_in)
        if layout == "channels_first":
            y = y.permute(0, 2, 3, 1)
        torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)

        y_ref.square().sum().backward()
        y.square().sum().backward()
        gx = x_in.grad.permute(0, 2, 3, 1) if layout == "channels_first" else x_in.grad
        torch.testing.assert_close(gx, x_ref.grad, atol=1e-4, rtol=1e-4)

        # sharded weight gradients
        tp, rank = comm.get_size("tp"), comm.get_rank("tp")
        n = hid // tp
        torch.testing.assert_close(mlp.fc1.weight.grad, serial.fc1.weight.grad[rank * n : (rank + 1) * n], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(mlp.fc2.weight.grad, serial.fc2.weight.grad[:, rank * n : (rank + 1) * n], atol=1e-4, rtol=1e-4)

    def test_channels_first_stays_in_the_autocast_dtype(self):
        """under autocast a channels-first MLP returns bf16, not fp32 (an fp32 bias must not promote the output)"""
        mlp = MLP(8, 16, 8, layout="channels_first")
        x = torch.randn(2, 8, 5, 7)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = mlp(x)
        self.assertEqual(y.dtype, torch.bfloat16)

    def test_default_is_channels_last(self):
        self.assertEqual(MLP(4, 8).fc1.layout, "channels_last")
        self.assertEqual(ColumnParallelLinear(4, 8).layout, "channels_last")
        self.assertEqual(RowParallelLinear(4, 8).layout, "channels_last")

    def test_invalid_layout(self):
        with self.assertRaises(ValueError):
            MLP(4, 8, layout="nchw")
