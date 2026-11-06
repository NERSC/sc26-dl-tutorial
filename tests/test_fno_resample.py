import math
import unittest

import torch
from parameterized import parameterized

from networks.fno import FourierNeuralOperator, FourierResample
from networks.helpers import resample_modes


def field(h, w, freqs):
    """a real field on an (h, w) grid made of plane waves with integer frequencies (fh, fw) and a mean"""
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    out = torch.full((h, w), 1.5)
    for i, (fh, fw) in enumerate(freqs):
        out = out + (1.0 / (1 + i)) * torch.cos(2 * math.pi * (fh * yy / h + fw * xx / w) + 0.3 * i)
    return out


class TestFourierResample(unittest.TestCase):
    @parameterized.expand([[64, 96, 2], [48, 48, 3], [63, 45, 3], [64, 64, 4]])
    def test_downsampling_samples_a_band_limited_field_exactly(self, h, w, f):
        """below the coarse Nyquist frequency, down(x) are the samples of x on the coarse sub-grid, and
        up(down(x)) gives x back (the sampling theorem)"""
        h2, w2 = h // f, w // f
        freqs = [(2, 1), (3, -2), (max(1, (h2 - 1) // 2 - 1), max(1, (w2 - 1) // 2 - 1))]
        x = field(h, w, freqs)[None, None]
        down = FourierResample((h, w), (h2, w2))(x)
        torch.testing.assert_close(down[0, 0], x[0, 0, ::f, ::f], atol=2e-5, rtol=0)
        up = FourierResample((h2, w2), (h, w))(down)
        torch.testing.assert_close(up, x, atol=2e-5, rtol=0)

    @parameterized.expand([[(64, 96), (32, 48)], [(33, 66), (17, 33)], [(30, 40), (10, 20)]])
    def test_values_are_preserved(self, big, small):
        """a constant stays that constant, and the mean square of a band-limited field is the same on both grids"""
        down = FourierResample(big, small)
        torch.testing.assert_close(down(torch.full((1, 1, *big), 3.0)), torch.full((1, 1, *small), 3.0), atol=1e-5, rtol=0)
        x = field(*big, [(2, 1), (1, 3)])[None, None]
        torch.testing.assert_close(down(x).square().mean(), x.square().mean(), atol=1e-4, rtol=1e-4)

    def test_down_of_up_is_the_identity_on_band_limited_fields(self):
        torch.manual_seed(0)
        small, big = (16, 24), (32, 48)
        z = FourierResample(big, small)(torch.randn(1, 3, *big))  # a band-limited field on the coarse grid
        torch.testing.assert_close(FourierResample(big, small)(FourierResample(small, big)(z)), z, atol=1e-5, rtol=1e-5)

    @parameterized.expand([[(8, 8)], [(7, 9)], [(1, 4)], [(2, 2)]])
    def test_modes_stay_strictly_below_nyquist(self, size):
        kh, kw = resample_modes(*size)
        self.assertLessEqual(kh, size[0])
        self.assertLessEqual(kw, size[1] // 2 + 1)
        if size[0] % 2 == 0:
            self.assertLess(kh, size[0])  # the Nyquist row is dropped, the kept count is odd and symmetric
            self.assertEqual(kh % 2, 1)
        if size[1] % 2 == 0:
            self.assertLessEqual(kw, size[1] // 2)  # no Nyquist column

    def test_real_output_dtype_and_gradient(self):
        x = torch.randn(2, 3, 32, 40, requires_grad=True)
        y = FourierResample((32, 40), (16, 20))(x)
        self.assertFalse(y.is_complex())
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertEqual(FourierResample((32, 40), (16, 20))(x.detach().to(torch.bfloat16)).dtype, torch.bfloat16)


class TestDownsamplingModel(unittest.TestCase):
    def test_the_default_model_is_unchanged(self):
        """downsample=1 has the parameters of the original FNO, with no widening layers and no resampling"""
        m = FourierNeuralOperator((32, 64), 4, 4, 16, 2, 8, 8)
        self.assertEqual(sorted({n.split(".")[0] for n, _ in m.named_parameters()}), ["blocks", "lift", "project"])
        self.assertEqual(m.latent_size, (32, 64))

    @parameterized.expand([[2], [4]])
    def test_blocks_run_on_the_latent_grid(self, f):
        m = FourierNeuralOperator((64, 128), 4, 3, 32, 2, 8, 8, downsample=f, lift_dim=8)
        self.assertEqual(m.latent_size, (64 // f, 128 // f))
        self.assertEqual(m.blocks[0].spectral.height, 64 // f)
        self.assertEqual(sorted({n.split(".")[0] for n, _ in m.named_parameters()}), ["blocks", "expand", "lift", "project", "reduce"])
        self.assertEqual(m.lift.fc2.out_channels, 8)
        self.assertEqual(m.expand.out_channels, 32)
        y = m(torch.randn(2, 4, 64, 128))
        self.assertEqual(tuple(y.shape), (2, 3, 64, 128))

    def test_lift_dim_defaults(self):
        self.assertEqual(FourierNeuralOperator((32, 32), 4, 4, 64, 1, 4, 4, downsample=2).lift.fc2.out_channels, 16)
        self.assertEqual(FourierNeuralOperator((32, 32), 4, 4, 64, 1, 4, 4).lift.fc2.out_channels, 64)

    def test_invalid_factor(self):
        for f in (0, 3, 5):
            with self.assertRaises(ValueError):
                FourierNeuralOperator((32, 64), 4, 4, 16, 1, 4, 4, downsample=f)

    def test_fewer_grid_points_in_the_blocks(self):
        """the activations of the blocks have f^2 times fewer points: record the shapes the blocks see"""
        shapes = []
        m = FourierNeuralOperator((64, 64), 4, 4, 16, 2, 8, 8, downsample=2, lift_dim=8)
        for b in m.blocks:
            b.register_forward_hook(lambda mod, i, o: shapes.append(tuple(o.shape)))
        m(torch.randn(1, 4, 64, 64))
        self.assertEqual(set(shapes), {(1, 16, 32, 32)})

    def test_compile_keeps_the_ffts_out_of_the_graph(self):
        torch._dynamo.reset()
        m = FourierNeuralOperator((32, 64), 4, 4, 16, 2, 6, 6, downsample=2, lift_dim=8)
        x = torch.randn(2, 4, 32, 64)
        ops = []

        def backend(gm, example_inputs):
            ops.extend(str(n.target) for n in gm.graph.nodes if n.op in ("call_function", "call_method", "call_module"))
            return gm.forward

        y = torch.compile(m, backend=backend)(x)
        self.assertTrue(ops)
        self.assertFalse([o for o in ops if "fft" in o.lower()], "an FFT is inside a compiled graph")
        torch.testing.assert_close(y, m(x), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
