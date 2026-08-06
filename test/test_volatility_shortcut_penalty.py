import sys
import unittest
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from train import (  # noqa: E402
    HeadFocusedRankingLoss,
    build_volatility_shortcut_index,
)


class VolatilityShortcutPenaltyTests(unittest.TestCase):
    def setUp(self):
        self.base_cfg = {
            "head_k": 2,
            "head_loss_weight": 1.0,
            "global_rank_weight": 0.2,
            "stability_weight": 0.05,
        }
        self.prediction = torch.tensor(
            [[-2.0, -1.0, 0.0, 1.0, 2.0]], requires_grad=True
        )
        self.target = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

    def test_absolute_correlation_penalizes_both_directions(self):
        criterion = HeadFocusedRankingLoss(
            {**self.base_cfg, "shortcut_correlation_weight": 0.05}
        )
        for shortcut in (
            torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]]),
            torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]]),
        ):
            _, components = criterion.forward_with_components(
                self.prediction, self.target, shortcut_index=shortcut
            )
            self.assertAlmostEqual(
                float(components["shortcut"].detach()), 0.05, places=6
            )

    def test_constant_index_is_safe_and_unpenalized(self):
        criterion = HeadFocusedRankingLoss(
            {**self.base_cfg, "shortcut_correlation_weight": 0.05}
        )
        total, components = criterion.forward_with_components(
            self.prediction,
            self.target,
            shortcut_index=torch.ones_like(self.prediction),
        )
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(float(components["shortcut"].detach()), 0.0)

    def test_penalty_does_not_change_existing_components(self):
        shortcut = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        disabled = HeadFocusedRankingLoss(
            {**self.base_cfg, "shortcut_correlation_weight": 0.0}
        )
        enabled = HeadFocusedRankingLoss(
            {**self.base_cfg, "shortcut_correlation_weight": 0.05}
        )
        _, old_components = disabled.forward_with_components(
            self.prediction, self.target, shortcut_index=shortcut
        )
        _, new_components = enabled.forward_with_components(
            self.prediction, self.target, shortcut_index=shortcut
        )
        for name in ("head", "global", "stability"):
            self.assertAlmostEqual(
                float(old_components[name].detach()),
                float(new_components[name].detach()),
                places=7,
            )

    def test_index_standardizes_features_before_equal_weight_average(self):
        sequences = torch.zeros(3, 2, 2)
        sequences[:, -1, 0] = torch.tensor([0.0, 1.0, 2.0])
        sequences[:, -1, 1] = torch.tensor([0.0, 100.0, 200.0])
        index = build_volatility_shortcut_index(
            sequences, ["STD5", "atr_14"], ["STD5", "atr_14"]
        )
        expected = torch.tensor([-1.2247449, 0.0, 1.2247449])
        self.assertTrue(torch.allclose(index, expected, atol=1e-6))
        self.assertFalse(index.requires_grad)

    def test_missing_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            build_volatility_shortcut_index(
                torch.zeros(3, 2, 1), ["STD5"], ["STD5", "atr_14"]
            )


if __name__ == "__main__":
    unittest.main()
