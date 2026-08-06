import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from train import WeightedRankingLoss  # noqa: E402
from utils import returns_to_relevance  # noqa: E402


class RankingTieTests(unittest.TestCase):
    def test_equal_returns_receive_equal_average_rank(self):
        actual = returns_to_relevance([0.10, 0.10, 0.05])
        np.testing.assert_allclose(actual, [2.5, 2.5, 1.0])

    def test_tie_rank_is_permutation_equivariant(self):
        original = returns_to_relevance([0.10, 0.10, 0.05])
        swapped = returns_to_relevance([0.10, 0.10, 0.05][::-1])[::-1]
        np.testing.assert_allclose(original, swapped)

    def test_tied_pair_has_no_pairwise_loss_or_gradient(self):
        criterion = WeightedRankingLoss()
        pred = torch.tensor([[2.0, -3.0]], requires_grad=True)
        relevance = torch.tensor([[1.5, 1.5]])
        weights = torch.ones_like(relevance)
        loss = criterion.pairwise_loss(pred, relevance, weights)
        loss.backward()
        torch.testing.assert_close(loss, torch.tensor(0.0))
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_non_tied_pair_prefers_higher_return_score(self):
        criterion = WeightedRankingLoss()
        relevance = torch.tensor([[2.0, 1.0]])
        weights = torch.ones_like(relevance)
        correct = criterion.pairwise_loss(
            torch.tensor([[1.0, 0.0]]), relevance, weights
        )
        wrong = criterion.pairwise_loss(
            torch.tensor([[0.0, 1.0]]), relevance, weights
        )
        self.assertLess(float(correct), float(wrong))


if __name__ == "__main__":
    unittest.main()
