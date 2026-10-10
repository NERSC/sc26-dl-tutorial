import torch
import torch.distributed as dist
from parameterized import parameterized

from utils import comm
from tests.helpers import DistributedTestCase
from distributed.helpers import init_params_for_shared_weights
from utils.loss import mse_loss
from networks.fno import FourierNeuralOperator as SerialFNO
from networks.fno_mp import FourierNeuralOperator as DistributedFNO


def shard_of(name, serial_param, dist_model):
    """
    This rank's shard of a serial parameter (a tensor, or a gradient), given the parameter name.
    Pointwise MLPs are sharded over tp (fc1 along the output, fc2 along the input channels),
    the spectral weight over the wavenumbers (sp1, sp2); the serial 1x1 convs carry two
    trailing singleton dims that the distributed linears do not have.
    """
    def pick(t):
        if t.dim() == 4 and t.shape[-2:] == (1, 1):
            t = t[:, :, 0, 0]
        tp, rtp = comm.get_size("tp"), comm.get_rank("tp")
        if ".mlp." in name or name.startswith(("lift.", "project.")):
            if ".fc1." in name:
                n = t.shape[0] // tp
                return t[rtp * n : (rtp + 1) * n]
            if name.endswith("fc2.weight"):
                n = t.shape[1] // tp
                return t[:, rtp * n : (rtp + 1) * n]
        if name.endswith("spectral.weight"):
            blk = dist_model.blocks[int(name.split(".")[1])].spectral
            h0 = sum(blk.forward_transform.kh_shapes[: comm.get_rank("sp1")])
            w0 = sum(blk.forward_transform.kw_shapes[: comm.get_rank("sp2")])
            return t[:, :, h0 : h0 + blk.modes_kh_local, w0 : w0 + blk.modes_kw_local]
        return t

    return pick(serial_param)


class TestFNOModel(DistributedTestCase):
    @parameterized.expand(
        [
            [(32, 64), 4, 3, 8, 2, None, None, 2.0],
            [(31, 62), 4, 3, 8, 2, 11, 9, 2.0],
            [(32, 64), 4, 3, 8, 1, 12, 17, 0.0],
        ]
    )
    def test_matches_serial(self, img_size, cin, cout, embed, depth, kh, kw, mlp_ratio):
        serial = SerialFNO(img_size, cin, cout, embed, depth, kh, kw, mlp_ratio).to(self.device)
        model = DistributedFNO(img_size, cin, cout, embed, depth, kh, kw, mlp_ratio).to(self.device)

        # same weights everywhere: broadcast the serial model, then take the shards
        with torch.no_grad():
            for p in serial.parameters():
                dist.broadcast(p, src=0)
        dist_params = dict(model.named_parameters())
        self.assertEqual(set(dist_params), set(dict(serial.named_parameters())))
        with torch.no_grad():
            for name, p in serial.named_parameters():
                dist_params[name].copy_(shard_of(name, p, model))

        # every rank takes its own shard of the input and returns its own shard of the output
        x = torch.randn(2, cin, *img_size, device=self.device)
        dist.broadcast(x, src=0)
        y_ref = serial(x)
        y = model(self._split(x))
        self.assertEqual(y.shape, self._split(y_ref).shape)
        torch.testing.assert_close(y, self._split(y_ref), atol=1e-4, rtol=1e-4)

        # gradients, after the reductions the DDP hook performs over each parameter's "reduce" groups
        y_ref.square().sum().backward()
        y.square().sum().backward()
        for name, p in model.named_parameters():
            for group in p.comm_metadata["reduce"]:
                dist.all_reduce(p.grad, group=comm.get_group(group))
            expected = shard_of(name, dict(serial.named_parameters())[name].grad, model)
            torch.testing.assert_close(p.grad, expected, atol=1e-3, rtol=1e-3, msg=lambda m: f"{name}: {m}")

    def test_sharded_loss(self):
        """PR #3's mse_loss on the shards equals the plain MSE of the full fields, with the same
        gradients, so no rank ever needs the full output"""
        img_size, cin, cout = (31, 62), 4, 3
        serial = SerialFNO(img_size, cin, cout, 8, 2, 11, 9).to(self.device)
        model = DistributedFNO(img_size, cin, cout, 8, 2, 11, 9).to(self.device)
        with torch.no_grad():
            for p in serial.parameters():
                dist.broadcast(p, src=0)
        dist_params = dict(model.named_parameters())
        with torch.no_grad():
            for name, p in serial.named_parameters():
                dist_params[name].copy_(shard_of(name, p, model))

        x = torch.randn(2, cin, *img_size, device=self.device)
        tar = torch.randn(2, cout, *img_size, device=self.device)
        dist.broadcast(x, src=0)
        dist.broadcast(tar, src=0)

        loss_ref = (serial(x) - tar).square().mean()
        loss = mse_loss(model(self._split(x)), self._split(tar), img_size)
        torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)

        loss_ref.backward()
        loss.backward()
        for name, p in model.named_parameters():
            for group in p.comm_metadata["reduce"]:
                dist.all_reduce(p.grad, group=comm.get_group(group))
            expected = shard_of(name, dict(serial.named_parameters())[name].grad, model)
            torch.testing.assert_close(p.grad, expected, atol=1e-4, rtol=1e-3, msg=lambda m: f"{name}: {m}")

    def test_activations_stay_bf16_under_autocast(self):
        """every block of the distributed FNO outputs bf16 under autocast, like the serial model. An fp32
        residual stream doubled the activation memory and ran a 16-GPU job out of memory."""
        model = DistributedFNO((32, 64), 4, 3, 8, 2, 12, 9).to(self.device)
        dtypes = []
        for blk in model.blocks:
            blk.register_forward_hook(lambda m, i, o: dtypes.append(o.dtype))
        x = self._split(torch.randn(2, 4, 32, 64, device=self.device))
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            y = model(x)
        self.assertEqual(set(dtypes), {torch.bfloat16})
        self.assertEqual(y.dtype, torch.bfloat16)

    def test_comm_metadata(self):
        model = DistributedFNO((32, 64), 4, 3, 8, 2, 12, 9)
        init_params_for_shared_weights(model)  # validates the metadata of every parameter

    def test_embed_dim_vs_groups(self):
        if comm.get_size("sp1") > 1:
            with self.assertRaises(ValueError):
                DistributedFNO((32, 64), 4, 3, embed_dim=comm.get_size("sp1") - 1)
