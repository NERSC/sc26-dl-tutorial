import math

import torch
import torch.distributed as dist

from utils import comm
from tests.helpers import DistributedTestCase
from distributed.helpers import init_params_for_shared_weights
from networks.fno import FourierNeuralOperator as SerialFNO
from networks.fno_mp import FourierNeuralOperator as DistributedFNO

CFG = dict(img_size=(32, 64), in_chans=4, out_chans=3, embed_dim=64, depth=2, modes_kh=12, modes_kw=9, mlp_ratio=2.0)


class TestFNOInit(DistributedTestCase):
    """The distributed initialization is statistically equivalent to the serial one, from global sizes.
    Not bitwise: every rank draws its own shard. The weight-copying equivalence tests cannot see this."""

    def setUp(self):
        torch.manual_seed(1000 + dist.get_rank())  # different draws on every rank, as in a real run
        self.serial = SerialFNO(**CFG).to(self.device)
        self.model = DistributedFNO(**CFG).to(self.device)
        init_params_for_shared_weights(self.model)  # what the trainer does

    def test_parameter_statistics_follow_the_global_rule(self):
        """sample std and mean of every parameter, within ~5 sigma of the sampling error for its size"""
        sparams = dict(self.serial.named_parameters())
        for name, p in self.model.named_parameters():
            n = p.numel() * (2 if p.is_complex() else 1)
            if p.is_complex():
                # complex normal, E|w|^2 = gain / C_in with the global input channels
                self.assertAlmostEqual(p.abs().pow(2).mean().item() * CFG["embed_dim"], 1.0, delta=5 / math.sqrt(n), msg=name)
                continue
            s = sparams[name]
            fan_in = s.shape[1] if s.dim() > 1 else sparams[name[:-5] + ".weight"].shape[1]  # global, from the serial layer
            if not name.endswith(".weight"):
                self.assertLessEqual(p.abs().max().item(), 1 / math.sqrt(fan_in) + 1e-6, msg=name)
            if n < 20:
                continue  # too few draws for a statistical statement; the bounds above still hold
            expected_std = 1 / math.sqrt(3 * fan_in)
            self.assertAlmostEqual(p.std().item() / expected_std, 1.0, delta=5 / math.sqrt(2 * n), msg=name)
            if name.endswith(".weight"):
                self.assertAlmostEqual(p.mean().item() / expected_std, 0.0, delta=5 / math.sqrt(n), msg=name)

    def test_replicated_parameters_are_identical_over_their_shared_groups(self):
        for name, p in self.model.named_parameters():
            for group in p.comm_metadata["shared"]:
                ref = p.detach().clone()
                dist.broadcast(ref, src=comm.get_root(group), group=comm.get_group(group))
                torch.testing.assert_close(p.detach(), ref, msg=lambda m: f"{name} over {group}: {m}")

    def test_signal_and_gradients_are_alive_and_comparable_to_serial(self):
        """Over many draws, the output and gradient scale at initialization of the distributed model has
        the distribution of the serial one. A single draw varies by more than 10x between seeds at this
        depth, so single networks cannot be compared. A constant std of 0.02 gave outputs and gradients
        of ~1e-13 here."""
        x = torch.randn(2, CFG["in_chans"], *CFG["img_size"], device=self.device)
        dist.broadcast(x, src=0)

        def scales(model, inp):
            y = model(inp)
            y.square().mean().backward()
            return y.std().item(), model.lift.fc1.weight.grad.norm().item()

        serial, distributed = [], []
        for draw in range(15):
            torch.manual_seed(5000 + 17 * draw)
            serial.append(scales(SerialFNO(**CFG).to(self.device), x))
            model = DistributedFNO(**CFG).to(self.device)
            init_params_for_shared_weights(model)
            distributed.append(scales(model, self._split(x)))

        for i, what in enumerate(("output std", "lift.fc1 gradient norm")):
            s = sorted(v[i] for v in serial)[len(serial) // 2]
            d = sorted(v[i] for v in distributed)[len(distributed) // 2]
            self.assertTrue(0.5 < d / s < 2.0, f"median {what}: distributed {d:.3e} vs serial {s:.3e}")
