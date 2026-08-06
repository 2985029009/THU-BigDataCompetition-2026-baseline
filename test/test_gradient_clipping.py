import sys
import unittest
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from config import config  # noqa: E402
from train import clip_gradients  # noqa: E402


class GradientClippingTests(unittest.TestCase):
    def test_large_gradient_is_clipped(self):
        parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        parameter.grad = torch.tensor([6.0, 8.0])
        stats = clip_gradients([parameter], 5.0)
        self.assertAlmostEqual(stats["grad_norm_before_clip"], 10.0)
        self.assertTrue(stats["gradient_clipped"])
        self.assertLessEqual(float(parameter.grad.norm()), 5.00001)

    def test_none_or_zero_disables_clipping(self):
        for threshold in (None, 0):
            parameter = torch.nn.Parameter(torch.tensor([1.0]))
            parameter.grad = torch.tensor([10.0])
            stats = clip_gradients([parameter], threshold)
            self.assertFalse(stats["gradient_clipped"])
            self.assertEqual(float(parameter.grad), 10.0)

    def test_default_configuration_enables_clipping(self):
        self.assertGreater(config["max_grad_norm"], 0)

    def test_non_finite_gradient_fails(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([float("nan")])
        with self.assertRaises(FloatingPointError):
            clip_gradients([parameter], 5.0)


if __name__ == "__main__":
    unittest.main()
