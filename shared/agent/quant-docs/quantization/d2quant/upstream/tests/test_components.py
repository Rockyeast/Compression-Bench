import unittest

import torch

from d2quant.quantization import DualScaleQuantizer2Bit, DualScaleQuantizer3Bit
from d2quant.quantization.dac import apply_mean_shift


class _RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, inputs):
        return inputs * self.weight


class ComponentTest(unittest.TestCase):
    def test_dual_scale_quantizers(self):
        weights = torch.randn(32, 128)
        for quantizer_class, bits in (
            (DualScaleQuantizer2Bit, 2),
            (DualScaleQuantizer3Bit, 3),
        ):
            quantizer = quantizer_class()
            quantizer.configure(bits=bits, num_iters=2)
            quantizer.find_params(weights)
            quantized = quantizer.quantize(weights)
            self.assertEqual(quantized.shape, weights.shape)
            self.assertTrue(torch.isfinite(quantized).all())

    def test_deviation_aware_mean_shift(self):
        layernorm = _RMSNorm(hidden_size=8)
        student = [torch.randn(16, 8)]
        teacher = [student[0] + 0.2]
        apply_mean_shift(layernorm, student, teacher)
        self.assertTrue(
            torch.allclose(
                layernorm.bias,
                torch.full((8,), 0.2),
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
