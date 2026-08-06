import sys
import unittest
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from config import config  # noqa: E402
from pit_data import apply_contest_score_label  # noqa: E402
from train import (  # noqa: E402
    HeadFocusedRankingLoss,
    build_ranking_criterion,
    calculate_ranking_metrics,
)


class Top5ObjectiveAlignmentTests(unittest.TestCase):
    def test_mainline_uses_head_focused_loss_and_top5_selection(self):
        self.assertEqual(config["ranking_loss_type"], "head_focused")
        self.assertEqual(
            config["model_selection_metric"], "top5_return"
        )
        self.assertEqual(config["label_mode"], "contest_score")
        self.assertIsInstance(build_ranking_criterion(config), HeadFocusedRankingLoss)

    def test_head_loss_rewards_separating_top_from_rest(self):
        criterion = HeadFocusedRankingLoss(
            {
                "head_k": 2,
                "head_loss_weight": 1.0,
                "global_rank_weight": 0.0,
                "stability_weight": 0.0,
            }
        )
        relevance = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        aligned = torch.tensor([[0.0, 0.1, 2.0, 3.0]])
        reversed_head = torch.tensor([[2.0, 3.0, 0.0, 0.1]])
        self.assertLess(
            float(criterion(aligned, relevance)),
            float(criterion(reversed_head, relevance)),
        )

    def test_metrics_expose_portfolio_return_and_excess_return(self):
        prediction = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        target = torch.tensor([[0.10, 0.05, 0.00, -0.05, -0.10]])
        mask = torch.ones_like(target)
        metrics = calculate_ranking_metrics(prediction, target, mask, k=2)
        self.assertAlmostEqual(metrics["top5_return"], 0.0)
        self.assertAlmostEqual(metrics["top5_excess_return"], 0.0)
        self.assertIn("ndcg_at_5", metrics)
        self.assertIn("top5_true_percentile", metrics)

    def test_contest_label_matches_first_to_fifth_future_open(self):
        import pandas as pd

        frame = pd.DataFrame(
            {
                "股票代码": ["000001"] * 7,
                "日期": pd.date_range("2026-01-01", periods=7),
                "开盘": [10.0, 11.0, 12.0, 13.0, 14.0, 16.5, 18.0],
            }
        )
        labeled = apply_contest_score_label(frame)
        self.assertAlmostEqual(
            labeled.loc[0, "label"], (16.5 - 11.0) / 11.0
        )


if __name__ == "__main__":
    unittest.main()
