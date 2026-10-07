import unittest

import torch
import torch.nn.functional as F
from parameterized import parameterized

from networks.fno import FourierNeuralOperator, FourierNeuralOperatorLayer
from networks.helpers import _truncate_modes, _zero_self_conjugate_imag, spectral_init_decay, spectral_wavenumbers

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def reference(layer, x):
    """full-tensor FNO layer built directly from torch.fft, no truncation helpers"""
    height, width = layer.height, layer.width
    kh, kw = layer.modes_kh, layer.modes_kw
    x, weight = x.cpu(), layer.weight.detach().cpu()
    spec = torch.fft.rfft2(x, norm="ortho")
    rows = list(range(kh - kh // 2)) + list(range(height - kh // 2, height))
    spec = spec[:, :, rows, :kw]
    spec = torch.einsum("bihw,iohw->bohw", spec, weight)
    full = torch.zeros(spec.shape[0], spec.shape[1], height, width // 2 + 1, dtype=spec.dtype)
    for i, r in enumerate(rows):
        full[:, :, r, :kw] = spec[:, :, i]
    return torch.fft.irfft2(full, s=(height, width), norm="ortho")


class TestFNOSerial(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(333)

    @parameterized.expand(
        [
            [dev, *cfg]
            for dev in DEVICES
            for cfg in [
                [32, 64, None, None],
                [31, 62, None, None],
                [32, 64, 12, 9],
                [33, 64, 11, 17],
                [32, 64, 1, 1],
            ]
        ]
    )
    def test_matches_reference(self, dev, height, width, kh, kw):
        layer = FourierNeuralOperatorLayer(3, 5, height, width, kh, kw).to(dev)
        x = torch.randn(2, 3, height, width, device=dev)
        y = layer(x)
        self.assertEqual(y.shape, (2, 5, height, width))
        self.assertFalse(y.is_complex())
        torch.testing.assert_close(y.cpu(), reference(layer, x), atol=1e-5, rtol=1e-5)

    def test_identity_weight(self):
        """all modes kept with unit diagonal weight is the identity"""
        layer = FourierNeuralOperatorLayer(4, 4, 16, 32)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[torch.arange(4), torch.arange(4)] = 1.0
        x = torch.randn(2, 4, 16, 32)
        torch.testing.assert_close(layer(x), x, atol=1e-5, rtol=1e-5)

    @parameterized.expand([[dev, *cfg] for dev in DEVICES for cfg in [[32, 64], [31, 62]]])
    def test_self_conjugate_imag_ignored(self, dev, height, width):
        """imaginary parts at DC and the Nyquist modes do not affect the output"""
        layer = FourierNeuralOperatorLayer(3, 3, height, width).to(dev)
        x = torch.randn(2, 3, height, width, device=dev)
        y0 = layer(x)

        # self-conjugate entries: kh in {0, height/2 (even height)}, kw in {0, width/2}.
        # Their weight indices are the same in the truncated layout (kh unpadded, in order)
        khs = [0] + ([height // 2] if height % 2 == 0 else [])
        kws = [0, width // 2]
        with torch.no_grad():
            for h in khs:
                for w in kws:
                    layer.weight[:, :, h, w] += 1j * torch.randn(3, 3, device=dev)
        torch.testing.assert_close(layer(x), y0, atol=1e-5, rtol=1e-5)

        # control: the same perturbation at a generic mode must change the output
        with torch.no_grad():
            layer.weight[:, :, 1, 1] += 1j * torch.randn(3, 3, device=dev)
        self.assertGreater((layer(x) - y0).abs().max().item(), 1e-3)

    @parameterized.expand([[64, 5], [62, 5], [64, 33], [63, 32]])
    def test_zero_self_conjugate_imag(self, width, m):
        x = torch.randn(2, 4, m, dtype=torch.complex64)
        y = _zero_self_conjugate_imag(x, width)
        zeroed = [0] + ([width // 2] if width % 2 == 0 and width // 2 < m else [])
        for k in range(m):
            if k in zeroed:
                self.assertTrue((y[..., k].imag == 0).all())
                torch.testing.assert_close(y[..., k].real, x[..., k].real)
            else:
                torch.testing.assert_close(y[..., k], x[..., k])

    @parameterized.expand([[dev] for dev in DEVICES])
    def test_compile_keeps_the_ffts_out_of_the_graph(self, dev):
        """Under torch.compile no FFT may appear in a compiled graph: with some torch versions inductor's FFT
        fallback kernels assert on the output strides (seen on the NERSC container at 4096^2). The pointwise
        layers still compile, and the result matches eager."""
        torch._dynamo.reset()
        model = FourierNeuralOperator((32, 64), 4, 4, 16, 2, 12, 17).to(dev)
        x = torch.randn(2, 4, 32, 64, device=dev)
        ops = []

        def backend(gm, example_inputs):
            ops.extend(str(n.target) for n in gm.graph.nodes if n.op in ("call_function", "call_method", "call_module"))
            return gm.forward

        y = torch.compile(model, backend=backend)(x)
        self.assertTrue(ops, "nothing was compiled")
        self.assertFalse([o for o in ops if "fft" in o.lower()], "an FFT is inside a compiled graph")
        torch.testing.assert_close(y, model(x), atol=1e-5, rtol=1e-5)
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(model.blocks[0].spectral.weight.grad).all())

    def test_pointwise_init_follows_the_fan_in_rule(self):
        """weights N(0, 1/(3 fan_in)), biases U(+-1/sqrt(fan_in)), with fan_in the layer's input channels"""
        model = FourierNeuralOperator((16, 32), 4, 3, 64, 2, 8, 8)
        for name, p in model.named_parameters():
            if p.is_complex():
                continue
            fan_in = p.shape[1] if name.endswith(".weight") else dict(model.named_parameters())[name[:-5] + ".weight"].shape[1]
            self.assertAlmostEqual(p.std().item() * (3 * fan_in) ** 0.5, 1.0, delta=0.2 if name.endswith(".weight") else 0.4, msg=name)
            if name.endswith(".bias"):
                self.assertLessEqual(p.abs().max().item(), fan_in**-0.5 + 1e-6, msg=name)

    @parameterized.expand([[5, 7], [6, 4], [12, 9], [1, 3]])
    def test_spectral_wavenumbers_follow_the_layout(self, mh, mw):
        """The wavenumber grid matches what the layer really produces: transform a plane wave of known frequency,
        keep the modes the way the layer does, and the index that lights up must carry that wavenumber."""
        height, width = 32, 40
        k = spectral_wavenumbers(mh, mw)
        self.assertEqual(tuple(k.shape), (mh, mw))
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        for fh in range(0, min(mh, height) // 2 + 1):
            for fw in range(0, mw):
                wave = torch.cos(2 * torch.pi * (fh * yy / height + fw * xx / width))
                full = torch.fft.rfft2(wave[None, None], norm="ortho")
                spec = _truncate_modes(full.narrow(-1, 0, mw), mh, dim=-2)[0, 0]
                # threshold relative to the untruncated spectrum, so a truncated line leaves nothing lit
                lit = (spec.abs() > 1e-3 * full.abs().max()).nonzero()
                if len(lit) == 0:
                    continue  # the line was truncated away (e.g. +kh of an even count keeps only -kh)
                expected = (fh**2 + fw**2) ** 0.5
                for i, j in lit.tolist():
                    self.assertAlmostEqual(k[i, j].item(), expected, places=4, msg=f"modes ({mh},{mw}) wave ({fh},{fw}) lit ({i},{j})")

    def test_spectral_wavenumbers_ranges_are_slices_of_the_global_grid(self):
        full = spectral_wavenumbers(12, 9)
        for (h0, h1), (w0, w1) in [((0, 3), (0, 3)), ((3, 6), (2, 5)), ((6, 12), (5, 9)), ((11, 12), (8, 9))]:
            torch.testing.assert_close(spectral_wavenumbers(12, 9, (h0, h1), (w0, w1)), full[h0:h1, w0:w1])

    def test_init_decay_follows_k0(self):
        """per-mode power of the initial weights is decay(k)^2 / C_in, and off by default"""
        torch.manual_seed(7)
        c = 64
        for k0 in (None, 8.0):
            layer = FourierNeuralOperatorLayer(c, c, 64, 128, 24, 20, init_k0=k0)
            power = layer.weight.detach().abs().pow(2).mean(dim=(0, 1))  # per mode, over the 64 x 64 weights
            k = spectral_wavenumbers(layer.modes_kh, layer.modes_kw)
            expected = (1.0 if k0 is None else spectral_init_decay(k, k0) ** 2) / c
            ratio = power / expected
            self.assertTrue(((ratio > 0.9) & (ratio < 1.1)).all(), f"k0={k0}: per-mode power off by {ratio.min():.2f}..{ratio.max():.2f}")

    def test_gradients(self):
        layer = FourierNeuralOperatorLayer(2, 2, 16, 16, 6, 5)
        x = torch.randn(1, 2, 16, 16, requires_grad=True)
        layer(x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(layer.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
