"""Serial and halo checks. Model equivalence lives in test_all.py.

Run alone for serial checks; use torchrun with SP1/SP2 for distributed checks.
"""

import os
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
import torch.distributed as dist
from torch import nn

from networks.convnext_mp import exchange_spatial_halo, exchange_width_stripes
from tests.helpers import copy_reference_parameters
from distributed.spatial_sharding import make_convnext_spatial_shard, make_grid_spatial_shard
from networks import convnext, convnext_mp
from utils import comm


def check_halo_exchange(exchange, device, kernel_size=7):
    """A diagonal impulse tests both forward ownership and backward routing."""
    halo = convnext.kernel_radius(kernel_size)
    sp1, sp2 = comm.get_size("sp1"), comm.get_size("sp2")
    extent = max(4, halo)
    shard = make_grid_spatial_shard(
        (extent * sp1, extent * sp2), sp1, comm.get_rank("sp1"), sp2, comm.get_rank("sp2"),
    )
    h, w = shard.core_slices
    full = torch.zeros(1, 1, *shard.global_shape, device=device)
    full[..., extent if sp1 > 1 else 2, extent if sp2 > 1 else 2] = 1
    local = full[..., h, w].clone().requires_grad_()
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
    padding = (0 if sp1 > 1 else halo, 0 if sp2 > 1 else halo)
    actual = torch.nn.functional.conv2d(exchange(local, halo), kernel, padding=padding)
    expected = torch.nn.functional.conv2d(full, kernel, padding=halo)[..., h, w]
    torch.testing.assert_close(actual, expected)
    seed = torch.zeros_like(full)
    seed[..., extent - 1 if sp1 > 1 else 1, extent - 1 if sp2 > 1 else 1] = 1
    (actual * seed[..., h, w]).sum().backward()
    expected_grad = torch.nn.functional.conv2d(seed, kernel, padding=halo)[..., h, w]
    torch.testing.assert_close(local.grad, expected_grad)


class Params(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class TestConvNeXt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.world_size = int(os.getenv("WORLD_SIZE", "1"))
        cls.sp1, cls.sp2 = int(os.getenv("SP1", "1")), int(os.getenv("SP2", "1"))
        cls.tp = int(os.getenv("TP", "1"))
        if cls.world_size % (cls.tp * cls.sp1 * cls.sp2):
            raise ValueError("WORLD_SIZE must be divisible by TP * SP1 * SP2")
        cls.dp = cls.world_size // (cls.tp * cls.sp1 * cls.sp2)
        cls.device = torch.device("cpu")
        if torch.cuda.is_available():
            cls.device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
            torch.cuda.set_device(cls.device)
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        if cls.world_size > 1:
            dist.init_process_group(
                backend="nccl" if cls.device.type == "cuda" else "gloo",
                timeout=timedelta(seconds=180),
            )
            comm.init_model_parallel_info(dp=cls.dp, tp=cls.tp, sp1=cls.sp1, sp2=cls.sp2)

    @classmethod
    def tearDownClass(cls):
        if cls.world_size > 1:
            dist.destroy_process_group()

    def params(self, **overrides):
        values = dict(
            img_size=(2 * (4 * self.sp1 - 1), 2 * (4 * self.sp2 - 1)),
            patch_size=2, embed_dim=8, depth=2, layer_scale_init=0.5,
            n_in_channels=2, n_out_channels=2, network="convnext",
        )
        values.update(overrides)
        return Params(**values)

    @staticmethod
    def make_shard(params):
        return make_convnext_spatial_shard(
            global_shape=params.img_size, patch_size=params.patch_size,
            sp1_size=comm.get_size("sp1"), sp1_rank=comm.get_rank("sp1"),
            sp2_size=comm.get_size("sp2"), sp2_rank=comm.get_rank("sp2"),
            order=params.get("order", "tp-sp1-sp2-dp"),
            kernel_size=params.get("kernel_size", 7), tp_size=comm.get_size("tp"),
        )

    @staticmethod
    def tile(x, shard):
        return x[..., shard.core_slices[0], shard.core_slices[1]]

    def test_field_contract_and_parameter_compatibility(self):
        params = self.params(n_out_channels=3)
        shard = self.make_shard(params)
        serial = convnext.ConvNeXt(params).to(self.device)
        spatial = convnext_mp.ConvNeXt(params, shard).to(self.device)
        copy_reference_parameters(serial, spatial, shard)
        spatial.load_state_dict(serial.state_dict(), strict=True)
        self.assertEqual(list(serial.state_dict()), list(spatial.state_dict()))
        x = torch.randn(2, 2, *params.img_size, device=self.device, requires_grad=True)
        y = serial(x)
        self.assertEqual(y.shape, (2, 3, *params.img_size))
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        for p in serial.parameters():
            self.assertIsNotNone(p.grad)
        for p in spatial.parameters():
            self.assertEqual(p.comm_metadata["reduce"], ["sp1-sp2"])
        self.assertTrue(all(m.p == 0 for m in serial.modules() if isinstance(m, nn.Dropout)))
        self.assertEqual(serial.blocks[0].mlp.fc1.out_features, 4 * params.embed_dim)
        self.assertTrue(all(s % params.patch_size == 0 for s in (*shard.h_splits, *shard.w_splits)))
        for splits in (shard.h_splits, shard.w_splits):
            if len(splits) > 1:
                self.assertGreater(len(set(splits)), 1)

    def test_invalid_configuration(self):
        for values, message in [
            ({"patch_size": 0}, "positive integer"),
            ({"patch_size": 1.5}, "positive integer"),
            ({"embed_dim": 0}, "positive integer"),
            ({"kernel_size": 0}, "positive integer"),
            ({"kernel_size": 4}, "odd"),
            ({"kernel_size": (3, 5)}, "positive integer"),
            ({"depth": False}, "positive integer"),
            ({"n_in_channels": 0}, "positive integer"),
            ({"img_size": (7, 8)}, "divisible"),
            ({"img_size": (8,)}, "pair"),
            ({"layer_scale_init": -1}, "nonnegative"),
            ({"layer_scale_init": float("nan")}, "finite"),
        ]:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                convnext.ConvNeXt(self.params(**values))
        with patch.object(comm, "get_size", side_effect=lambda g: 3 if g == "tp" else 1), \
             patch.object(comm, "get_rank", return_value=0):
            params = self.params()
            with self.assertRaisesRegex(ValueError, "use tp=1"):
                self.make_shard(params)
            with self.assertRaisesRegex(ValueError, "use tp=1"):
                convnext_mp.ConvNeXt(params, None)

    def test_rejects_bad_tiles_before_communication(self):
        # Mock a mesh so this check also runs in a serial test process.
        with patch.object(comm, "get_size", side_effect=lambda g: 2 if g == "sp2" else 1), \
             patch.object(comm, "get_rank", return_value=0), \
             patch("distributed.mappings._exchange_halo_edges", side_effect=AssertionError("unexpected exchange")):
            params = self.params(img_size=(8, 14))
            shard = self.make_shard(params)
            bad = replace(shard, w_splits=(7, 7))
            with self.assertRaisesRegex(ValueError, "patch-aligned"):
                convnext_mp.ConvNeXt(params, bad)
            # Rank zero has three cells, but its neighbor only has two.
            with self.assertRaisesRegex(ValueError, "at least 3 feature cells"):
                self.make_shard(self.params(img_size=(8, 10)))

    def test_singleton_axes_and_activation_dtype(self):
        with patch.object(comm, "get_size", return_value=1), \
             patch("distributed.mappings._exchange_halo_edges", side_effect=AssertionError("unexpected exchange")):
            x = torch.randn(1, 2, 1, 1, device=self.device)
            self.assertIs(exchange_width_stripes(x, 3), x)
            self.assertIs(exchange_spatial_halo(x, 3), x)
            serial = nn.Conv2d(2, 2, 7, padding=3, groups=2).to(self.device)
            spatial = convnext_mp.HaloDepthwiseConv2d(2).to(self.device)
            spatial.load_state_dict(serial.state_dict())
            torch.testing.assert_close(serial(x), spatial(x))
        dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        block = convnext.ConvNeXtBlock(4).to(self.device)
        norm = convnext.LayerNorm2d(4).to(self.device)
        x = torch.randn(2, 4, 3, 5, dtype=dtype, device=self.device)
        with torch.autocast(self.device.type, dtype=dtype):
            self.assertEqual(block(x).dtype, dtype)
            self.assertEqual(norm(x).dtype, dtype)

    def test_convolution_values_and_gradients(self):
        # Uneven tiles of 3 or 4 cells: both three-cell gradient bands overlap.
        shape = (4 * self.sp1 - 1, 4 * self.sp2 - 1)
        shard = make_grid_spatial_shard(shape, self.sp1, comm.get_rank("sp1"), self.sp2, comm.get_rank("sp2"))
        torch.manual_seed(40)
        full = torch.randn(2, 2, *shape, device=self.device, dtype=torch.float64)
        for kernel_size in (1, 3, 7):
            reference = full.clone().requires_grad_()
            local = self.tile(full, shard).clone().requires_grad_()
            serial = nn.Conv2d(2, 2, kernel_size, padding=kernel_size // 2, groups=2).to(self.device).double()
            spatial = convnext_mp.HaloDepthwiseConv2d(2, kernel_size).to(self.device).double()
            spatial.load_state_dict(serial.state_dict())
            actual, expected = spatial(local), serial(reference)
            torch.testing.assert_close(actual, self.tile(expected, shard), atol=1e-11, rtol=1e-11)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(local.grad, self.tile(reference.grad, shard), atol=1e-10, rtol=1e-10)
            for p, ref in zip(spatial.parameters(), serial.parameters()):
                if self.sp1 * self.sp2 > 1:
                    dist.all_reduce(p.grad, group=comm.get_group("sp1-sp2"))
                torch.testing.assert_close(p.grad, ref.grad, atol=1e-9, rtol=1e-10)

    def test_corner_dependencies(self):
        check_halo_exchange(exchange_spatial_halo, self.device)

    def test_fullgraph_serial_forward_backward(self):
        model = convnext.ConvNeXt(self.params()).to(self.device)
        compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
        x = torch.randn(1, 2, *model.img_size, device=self.device, requires_grad=True)
        compiled(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())



if __name__ == "__main__":
    unittest.main()
