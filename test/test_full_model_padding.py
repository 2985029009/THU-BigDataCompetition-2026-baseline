import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


SRC_DIR = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from model import INFO_TCN_iTransformer, TemporalConvNet  # noqa: E402


def small_config(norm_type="layernorm"):
    return {
        "d_model": 8,
        "sequence_length": 4,
        "dropout": 0.0,
        "model_variant": "hybrid",
        "tcn_channels": 8,
        "tcn_kernel_size": 2,
        "tcn_num_layers": 2,
        "tcn_dropout": 0.0,
        "tcn_norm_type": norm_type,
        "it_nhead": 2,
        "it_num_layers": 1,
        "it_dim_feedforward": 16,
        "it_variable_identity_encoding": "learned",
        "fusion_dropout": 0.0,
        "nhead": 2,
    }


class FullModelPaddingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = INFO_TCN_iTransformer(
            input_dim=3, config=small_config(), num_stocks=4
        )
        self.valid = torch.randn(1, 2, 4, 3)

    def _inputs(self, padding):
        src = torch.cat([self.valid, padding], dim=1)
        mask = torch.tensor(
            [[False, False] + [True] * padding.shape[1]], dtype=torch.bool
        )
        return src, mask

    def test_eval_output_ignores_padding_content(self):
        self.model.eval()
        a, mask = self._inputs(torch.zeros(1, 1, 4, 3))
        b, _ = self._inputs(torch.full((1, 1, 4, 3), 1000.0))
        with torch.no_grad():
            torch.testing.assert_close(
                self.model(a, mask)[:, :2], self.model(b, mask)[:, :2]
            )

    def test_train_output_ignores_padding_count(self):
        self.model.train()
        a, mask_a = self._inputs(torch.zeros(1, 1, 4, 3))
        b, mask_b = self._inputs(torch.randn(1, 2, 4, 3))
        torch.testing.assert_close(
            self.model(a, mask_a)[:, :2], self.model(b, mask_b)[:, :2]
        )

    def test_valid_input_gradient_ignores_padding_content(self):
        model_a = copy.deepcopy(self.model).train()
        model_b = copy.deepcopy(self.model).train()
        valid_a = self.valid.clone().requires_grad_(True)
        valid_b = self.valid.clone().requires_grad_(True)
        mask = torch.tensor([[False, False, True]])
        src_a = torch.cat([valid_a, torch.zeros(1, 1, 4, 3)], dim=1)
        src_b = torch.cat([valid_b, torch.full((1, 1, 4, 3), 99.0)], dim=1)
        model_a(src_a, mask)[:, :2].sum().backward()
        model_b(src_b, mask)[:, :2].sum().backward()
        torch.testing.assert_close(valid_a.grad, valid_b.grad)

    def test_mainline_contains_no_batchnorm(self):
        self.assertFalse(
            any(isinstance(module, nn.BatchNorm1d) for module in self.model.modules())
        )

    def test_historical_config_defaults_to_batchnorm(self):
        historical = small_config()
        historical.pop("tcn_norm_type")
        branch = TemporalConvNet(3, 8, historical)
        self.assertTrue(
            any(isinstance(module, nn.BatchNorm1d) for module in branch.modules())
        )


if __name__ == "__main__":
    unittest.main()
