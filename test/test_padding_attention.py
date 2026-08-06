import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from model import CrossStockAttention  # noqa: E402


class PaddingAttentionTests(unittest.TestCase):
    def test_padding_values_do_not_change_valid_stock_outputs(self):
        torch.manual_seed(42)
        attention = CrossStockAttention(d_model=8, nhead=2, dropout=0.0).eval()

        valid = torch.randn(1, 2, 8)
        padded_a = torch.zeros(1, 1, 8)
        padded_b = torch.full((1, 1, 8), 1000.0)
        mask = torch.tensor([[False, False, True]])

        with torch.no_grad():
            output_a = attention(torch.cat([valid, padded_a], dim=1), mask)
            output_b = attention(torch.cat([valid, padded_b], dim=1), mask)

        torch.testing.assert_close(output_a[:, :2], output_b[:, :2])

    def test_attention_remains_backward_compatible_without_padding(self):
        torch.manual_seed(42)
        attention = CrossStockAttention(d_model=8, nhead=2, dropout=0.0).eval()
        stocks = torch.randn(1, 3, 8)

        with torch.no_grad():
            output = attention(stocks)

        self.assertEqual(output.shape, stocks.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_padding_mask_shape_is_validated(self):
        attention = CrossStockAttention(d_model=8, nhead=2, dropout=0.0).eval()
        stocks = torch.randn(1, 3, 8)

        with self.assertRaisesRegex(ValueError, "padding_mask"):
            attention(stocks, torch.zeros(1, 2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
