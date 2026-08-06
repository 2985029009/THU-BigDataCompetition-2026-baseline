import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


SRC_DIR = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from train import (  # noqa: E402
    WeightedRankingLoss,
    summarize_target_distributions,
)


class ListwiseNormalizationTests(unittest.TestCase):
    def test_weighted_target_distribution_is_renormalized(self):
        loss_fn = WeightedRankingLoss(
            temperature=1.0,
            k=2,
            weight_factor=2.0,
            base_weight=1.0,
        )
        y_pred = torch.tensor([[0.2, -0.1, 0.4, 0.0]], dtype=torch.float32)
        y_true = torch.tensor([[1.0, 2.0, 4.0, 3.0]], dtype=torch.float32)
        weights = torch.tensor([[1.0, 1.0, 2.0, 2.0]], dtype=torch.float32)

        target_probs = loss_fn.target_distribution(y_true)
        expected_targets = target_probs * weights
        expected_targets = expected_targets / expected_targets.sum(
            dim=1, keepdim=True
        )
        expected = -(
            expected_targets * F.log_softmax(y_pred, dim=1)
        ).sum(dim=1).mean()

        actual = loss_fn.listwise_loss(y_pred, y_true, weights)
        torch.testing.assert_close(actual, expected)

    def test_loss_is_not_divided_by_stock_count(self):
        loss_fn = WeightedRankingLoss(k=5, weight_factor=2.0, base_weight=1.0)
        y_pred = torch.zeros((1, 300), dtype=torch.float32)
        y_true = torch.arange(300, dtype=torch.float32).unsqueeze(0)
        weights = torch.ones_like(y_true)
        weights[:, -5:] = 2.0

        actual = loss_fn.listwise_loss(y_pred, y_true, weights)
        expected_uniform_cross_entropy = torch.log(torch.tensor(300.0))
        torch.testing.assert_close(
            actual, expected_uniform_cross_entropy, rtol=1e-5, atol=1e-5
        )

    def test_normalized_rank_target_is_not_top_one_degenerate(self):
        loss_fn = WeightedRankingLoss(
            target_type="normalized_rank_softmax", temperature=1.0
        )
        relevance = torch.arange(1, 301, dtype=torch.float32).unsqueeze(0)
        probs = loss_fn.target_distribution(relevance)

        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)
        self.assertLess(float(probs[:, -5:].sum()), 0.10)

    def test_target_distribution_is_invariant_to_rank_scale_and_shift(self):
        loss_fn = WeightedRankingLoss(target_type="normalized_rank_softmax")
        relevance = torch.tensor([[1.0, 2.5, 2.5, 4.0]])
        torch.testing.assert_close(
            loss_fn.target_distribution(relevance),
            loss_fn.target_distribution(relevance * 10.0 + 7.0),
        )

    def test_extreme_logits_keep_finite_loss_and_gradient(self):
        loss_fn = WeightedRankingLoss(target_type="normalized_rank_softmax")
        pred = torch.tensor([[1e6, -1e6, 0.0]], requires_grad=True)
        relevance = torch.tensor([[3.0, 1.0, 2.0]])
        loss = loss_fn(pred, relevance)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_distribution_diagnostics_report_size_and_concentration_ranges(self):
        loss_fn = WeightedRankingLoss(target_type="normalized_rank_softmax")
        summary = summarize_target_distributions(
            [
                torch.arange(1, 101, dtype=torch.float32),
                torch.arange(1, 301, dtype=torch.float32),
            ],
            loss_fn,
        )
        self.assertEqual(summary["num_cross_sections"], 2)
        self.assertEqual(summary["num_stocks"]["min"], 100)
        self.assertEqual(summary["num_stocks"]["max"], 300)
        self.assertIn("top10_mass", summary)
        self.assertIn("entropy", summary)


if __name__ == "__main__":
    unittest.main()
