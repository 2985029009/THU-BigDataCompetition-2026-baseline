import sys
import unittest
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from config import config  # noqa: E402
from train import (  # noqa: E402
    RankNetStabilityLoss,
    calculate_ranking_metrics,
    summarize_daily_ics,
    train_ranking_model,
)


class LossDiagnosticTests(unittest.TestCase):
    def test_mainline_capacity_is_restored(self):
        self.assertEqual(config["d_model"], 256)

    def test_component_sum_matches_total_and_preserves_backward(self):
        criterion = RankNetStabilityLoss(
            {
                "top5_weight": 2.0,
                "pairwise_weight": 1.0,
                "base_weight": 1.0,
                "listwise_temperature": 1.0,
                "listwise_target_type": "normalized_rank_softmax",
            }
        )
        prediction = torch.tensor(
            [[3.0, 2.0, 1.0, 0.0, -1.0]], requires_grad=True
        )
        relevance = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        total, components = criterion.forward_with_components(
            prediction, relevance
        )
        torch.testing.assert_close(total, sum(components.values()))

        expected_prediction = prediction.detach().clone().requires_grad_(True)
        expected_loss = criterion(expected_prediction, relevance)
        expected_loss.backward()
        total.backward()
        torch.testing.assert_close(
            prediction.grad, expected_prediction.grad
        )

    def test_ranking_metrics_omit_prediction_cross_section_std(self):
        prediction = torch.tensor([[1.0, 3.0, 1000.0]])
        target = torch.tensor([[0.1, 0.2, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        metrics = calculate_ranking_metrics(
            prediction, target, mask, k=2
        )
        self.assertNotIn("prediction_cross_section_std", metrics)

    def test_daily_ic_summary_omits_positive_ratio(self):
        metrics = summarize_daily_ics([0.2, 0.0, -0.1, 0.3])
        self.assertNotIn("ic_positive_ratio", metrics)
        self.assertAlmostEqual(metrics["mean_ic"], 0.1)

    def test_training_step_omits_removed_diagnostics(self):
        class TinyRanker(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, src, stock_padding_mask=None):
                return self.linear(src.mean(dim=2)).squeeze(-1)

        model = TinyRanker()
        criterion = RankNetStabilityLoss(
            {
                "top5_weight": 2.0,
                "pairwise_weight": 1.0,
                "base_weight": 1.0,
                "listwise_temperature": 1.0,
                "listwise_target_type": "normalized_rank_softmax",
            }
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = {
            "sequences": torch.arange(10, dtype=torch.float32).reshape(
                1, 5, 2, 1
            ),
            "targets": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            "relevance": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
            "masks": torch.ones(1, 5),
        }
        loss, metrics = train_ranking_model(
            model,
            [batch],
            criterion,
            optimizer,
            torch.device("cpu"),
            epoch=0,
            writer=None,
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertNotIn("listwise_grad_norm", metrics)
        self.assertNotIn("pairwise_grad_norm", metrics)
        self.assertNotIn("stability_grad_norm", metrics)
        self.assertNotIn("prediction_cross_section_std", metrics)


if __name__ == "__main__":
    unittest.main()
