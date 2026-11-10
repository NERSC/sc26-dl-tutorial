import unittest

import torch
from parameterized import parameterized

from networks.fno import FourierNeuralOperatorLayer
from networks.helpers import _zero_self_conjugate_imag


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def reference(layer, x):
    """One spectral layer from torch.fft, using the same inverse path as the layer.

    The inverse is ifft along height, then irfft along width. irfft2 does not match
    that path closely enough on CUDA to be the reference.
    """
    height, width = layer.height, layer.width
    kh, kw = layer.modes_kh, layer.modes_kw
    weight = layer.weight.detach()
    spec = torch.fft.rfft2(x, norm="ortho")
    rows = list(range(kh - kh // 2)) + list(range(height - kh // 2, height))
    spec = spec[:, :, rows, :kw]
    spec = torch.einsum("bihw,iohw->bohw", spec, weight)
    full = spec.new_zeros(spec.shape[0], spec.shape[1], height, width // 2 + 1)
    for index, row in enumerate(rows):
        full[:, :, row, :kw] = spec[:, :, index]
    full = torch.fft.ifft(full, n=height, dim=-2, norm="ortho")
    full = _zero_self_conjugate_imag(full, width)
    return torch.fft.irfft(full, n=width, dim=-1, norm="ortho")


class TestFNOSerial(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(333)

    @parameterized.expand([[dev] for dev in DEVICES])
    def test_matches_reference(self, dev):
        layer = FourierNeuralOperatorLayer(3, 5, 32, 64, 12, 9).to(dev)
        x = torch.randn(2, 3, 32, 64, device=dev)
        y = layer(x)
        self.assertEqual(y.shape, (2, 5, 32, 64))
        self.assertFalse(y.is_complex())
        torch.testing.assert_close(y, reference(layer, x), atol=1e-5, rtol=1e-5)

    def test_identity_weight(self):
        """Keeping every mode with a unit diagonal weight is the identity."""
        layer = FourierNeuralOperatorLayer(4, 4, 16, 32)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[torch.arange(4), torch.arange(4)] = 1.0
        x = torch.randn(2, 4, 16, 32)
        torch.testing.assert_close(layer(x), x, atol=1e-5, rtol=1e-5)

    @parameterized.expand([[dev] for dev in DEVICES])
    def test_self_conjugate_imag_ignored(self, dev):
        """Imaginary parts at DC and Nyquist do not affect the output."""
        layer = FourierNeuralOperatorLayer(3, 3, 32, 64).to(dev)
        x = torch.randn(2, 3, 32, 64, device=dev)
        y0 = layer(x)

        with torch.no_grad():
            for height_bin in (0, 16):
                for width_bin in (0, 32):
                    layer.weight[:, :, height_bin, width_bin] += 1j * torch.randn(
                        3, 3, device=dev
                    )
        torch.testing.assert_close(layer(x), y0, atol=1e-5, rtol=1e-5)

        with torch.no_grad():
            layer.weight[:, :, 1, 1] += 1j * torch.randn(3, 3, device=dev)
        self.assertGreater((layer(x) - y0).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
