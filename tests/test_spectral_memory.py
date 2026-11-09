import unittest

import torch
import torch.nn.functional as F
from parameterized import parameterized

from networks.fno import FourierNeuralOperatorLayer, FourierResample
from networks.helpers import _zero_self_conjugate_imag


def saved_bytes(fn, *inputs, params=()):
    """bytes of distinct storage that autograd keeps for the backward pass of fn(*inputs), not counting the
    parameters, which exist anyway"""
    ptrs, total = {p.untyped_storage().data_ptr() for p in params}, 0

    def pack(t):
        nonlocal total
        if t.untyped_storage().data_ptr() not in ptrs:
            ptrs.add(t.untyped_storage().data_ptr())
            total += t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn(*inputs)
    return total, out


class TestSpectralGradients(unittest.TestCase):
    """The complex-free mask is the same map as the torch.complex formulation, with the same gradients."""

    @parameterized.expand([[-1, 64, 33], [-1, 63, 32], [3, 64, 33], [2, 8, 9], [-2, 8, 9]])
    def test_zero_self_conjugate_imag_matches_the_complex_form(self, dim, n, size):
        shape = [2, 5, 9, 33] if dim in (-1, 3) else [2, 5, 9, 33]
        shape[dim] = size
        x = torch.randn(*shape, dtype=torch.complex64, requires_grad=True)
        scale = torch.ones(size)
        scale[0] = 0.0
        if n % 2 == 0 and n // 2 < size:
            scale[n // 2] = 0.0
        view = [1] * x.ndim
        view[dim] = size
        ref = torch.complex(x.real, x.imag * scale.reshape(view))  # the formulation of torch_harmonics
        out = _zero_self_conjugate_imag(x, n, dim=dim)
        torch.testing.assert_close(out, ref)
        g = torch.randn_like(ref)
        (gx,) = torch.autograd.grad(ref, x, g)
        (gx2,) = torch.autograd.grad(out, x, g)
        torch.testing.assert_close(gx2, gx)

    def test_layer_gradients_match_plain_autograd(self):
        """the spectral layer against the same computation written with plain torch ops"""
        torch.manual_seed(0)
        layer = FourierNeuralOperatorLayer(3, 4, 16, 24, 6, 7)
        x = torch.randn(2, 3, 16, 24, requires_grad=True)
        y = layer(x)
        (gx, gw) = torch.autograd.grad(y.square().sum(), (x, layer.weight))

        from networks.helpers import _pad_modes, _truncate_modes

        x2 = x.detach().clone().requires_grad_(True)
        w2 = layer.weight.detach().clone().requires_grad_(True)
        s = torch.fft.rfft2(x2, s=(16, 24), norm="ortho")
        s = _truncate_modes(s.narrow(-1, 0, 7), 6, dim=-2)
        s = torch.einsum("bihw,iohw->bohw", s, w2)
        s = F.pad(_pad_modes(s, 6, 16, dim=-2), (0, 24 // 2 + 1 - 7))
        s = torch.fft.ifft(s, n=16, dim=-2, norm="ortho")
        scale = torch.ones(s.shape[-1])
        scale[0] = 0.0
        scale[12] = 0.0
        s = torch.complex(s.real, s.imag * scale)
        ref = torch.fft.irfft(s, n=24, dim=-1, norm="ortho")
        torch.testing.assert_close(y, ref, atol=1e-6, rtol=1e-5)
        (rx, rw) = torch.autograd.grad(ref.square().sum(), (x2, w2))
        torch.testing.assert_close(gx, rx, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(gw, rw, atol=1e-4, rtol=1e-4)


class TestSpectralLayersKeepOneCopyAtFullResolution(unittest.TestCase):
    """What autograd keeps for the backward pass of a spectral layer is the (fp32) input of rfft2, the truncated
    spectrum and the weight. rfft2 keeps its input although its backward only needs the shape; we accept that one
    copy for simplicity. torch.complex used to keep both of its inputs on top of that, the whole complex field once
    more, and these tests fail if a layer ever keeps such an additional full-resolution tensor again."""

    def test_spectral_layer(self):
        h, w, c = 128, 128, 8
        layer = FourierNeuralOperatorLayer(c, c, h, w, 16, 16)
        x = torch.randn(2, c, h, w, requires_grad=True)
        kept, _ = saved_bytes(layer, x, params=layer.parameters())
        field = x.numel() * 4
        # the rfft2 input (1 x the field) and the truncated spectrum, 2 x 8 x 16 x 16 complex = 32 KB of a 1 MB
        # field (3%); the torch.complex mask kept the complex field once more, about 2 x the input in total
        self.assertLess(kept, 1.05 * field, f"the layer keeps {kept / field:.2f} x the input field for backward")

    @parameterized.expand([[(128, 128), (64, 64)], [(64, 64), (128, 128)]])
    def test_resampling(self, in_size, out_size):
        x = torch.randn(2, 8, *in_size, requires_grad=True)
        kept, _ = saved_bytes(FourierResample(in_size, out_size), x)
        field = x.numel() * 4
        self.assertLess(kept, field + 0.02 * max(field, 2 * 8 * out_size[0] * out_size[1] * 4))


if __name__ == "__main__":
    unittest.main()
