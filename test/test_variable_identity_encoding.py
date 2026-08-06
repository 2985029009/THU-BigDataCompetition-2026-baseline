import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from model import iTransformerBranch  # noqa: E402


class CaptureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach().clone()
        return x


class VariableIdentityEncodingTests(unittest.TestCase):
    def _build_branch(self):
        return iTransformerBranch(
            input_dim=4,
            seq_len=6,
            d_model=8,
            config={
                "it_nhead": 2,
                "it_num_layers": 1,
                "it_dim_feedforward": 16,
                "it_variable_identity_encoding": "learned",
                "dropout": 0.0,
            },
        )

    def test_each_variable_has_a_learned_identity_vector(self):
        branch = self._build_branch()

        self.assertIsInstance(branch.variable_identity_embedding, nn.Embedding)
        self.assertEqual(
            tuple(branch.variable_identity_embedding.weight.shape),
            (4, 8),
        )
        self.assertTrue(branch.variable_identity_embedding.weight.requires_grad)

    def test_identity_is_added_before_variable_attention(self):
        torch.manual_seed(42)
        branch = self._build_branch().eval()
        capture = CaptureEncoder()
        branch.encoder = capture
        common_history = torch.randn(2, 6, 1)
        inputs = common_history.expand(-1, -1, 4).clone()

        with torch.no_grad():
            branch(inputs)

        identities = branch.variable_identity_embedding.weight
        torch.testing.assert_close(
            capture.last_input[:, 1] - capture.last_input[:, 0],
            (identities[1] - identities[0]).expand(2, -1),
        )

    def test_identity_embedding_receives_gradients(self):
        torch.manual_seed(42)
        branch = self._build_branch()
        branch(torch.randn(3, 6, 4)).square().sum().backward()

        gradient = branch.variable_identity_embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_historical_config_without_field_keeps_old_state_dict_shape(self):
        branch = iTransformerBranch(
            input_dim=4,
            seq_len=6,
            d_model=8,
            config={
                "it_nhead": 2,
                "it_num_layers": 1,
                "it_dim_feedforward": 16,
                "dropout": 0.0,
            },
        )

        self.assertIsNone(branch.variable_identity_embedding)
        self.assertNotIn(
            "variable_identity_embedding.weight",
            branch.state_dict(),
        )


if __name__ == "__main__":
    unittest.main()
