import torch
import torch.distributed as dist
from parameterized import parameterized

from utils import comm
from tests.helpers import DistributedTestCase
from tests.test_fno_model import shard_of
from distributed.helpers import init_params_for_shared_weights
from distributed.mappings import init_ddp_model_and_reduction_hooks
from networks.fno import FourierNeuralOperator as SerialFNO
from networks.fno_mp import FourierNeuralOperator as DistributedFNO


class TestFNODDP(DistributedTestCase):
    """The FNO under the real DDP wrapper and gradient-reduction hooks, with dp > 1"""

    @parameterized.expand([[(32, 64), 8, 2, 12, 9, 2.0], [(31, 62), 8, 1, None, None, 0.0]])
    def test_ddp_grads_match_serial(self, img_size, embed, depth, kh, kw, mlp_ratio):
        dp = comm.get_size("dp")
        self.assertGreater(dp, 1, "run with a world size larger than tp*sp1*sp2")

        serial = SerialFNO(img_size, 4, 3, embed, depth, kh, kw, mlp_ratio).to(self.device)
        model = DistributedFNO(img_size, 4, 3, embed, depth, kh, kw, mlp_ratio).to(self.device)
        with torch.no_grad():
            for p in serial.parameters():
                dist.broadcast(p, src=0)
        dist_params = dict(model.named_parameters())
        with torch.no_grad():
            for name, p in serial.named_parameters():
                dist_params[name].copy_(shard_of(name, p, model))

        # every dp rank gets its own batch, all ranks can regenerate all of them
        batches = [
            torch.randn(2, 4, *img_size, generator=torch.Generator().manual_seed(100 + i)).to(self.device)
            for i in range(dp)
        ]

        # reference: gradient of the mean over dp of the per-batch losses
        for x in batches:
            (serial(x).square().sum() / dp).backward()

        init_params_for_shared_weights(model)
        ddp = init_ddp_model_and_reduction_hooks(model, device_ids=None, output_device=None)
        ddp(self._split(batches[comm.get_rank("dp")])).square().sum().backward()

        for name, p in model.named_parameters():
            expected = shard_of(name, dict(serial.named_parameters())[name].grad, model)
            torch.testing.assert_close(p.grad, expected, atol=1e-3, rtol=1e-3, msg=lambda m: f"{name}: {m}")
