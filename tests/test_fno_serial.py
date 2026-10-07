import unittest

import torch
import torch.nn.functional as F
from parameterized import parameterized

from networks.fno import FourierNeuralOperator, FourierNeuralOperatorLayer
from networks.helpers import _zero_self_conjugate_imag

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
    def test_compile_fullgraph(self, dev):
        layer = FourierNeuralOperatorLayer(4, 4, 32, 64, 12, 17).to(dev)
        x = torch.randn(2, 4, 32, 64, device=dev, requires_grad=True)
        y = torch.compile(layer, fullgraph=True)(x)
        torch.testing.assert_close(y, layer(x), atol=1e-5, rtol=1e-5)
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(layer.weight.grad).all())

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

    def test_gradients(self):
        layer = FourierNeuralOperatorLayer(2, 2, 16, 16, 6, 5)
        x = torch.randn(1, 2, 16, 16, requires_grad=True)
        layer(x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(layer.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
