import sys
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from train import split_dates_train_val_test  # noqa: E402


class IndependentGap2SplitTests(unittest.TestCase):
    def test_independent_gap2_preserves_requested_segment_lengths(self):
        dates = pd.bdate_range("2024-01-01", periods=512)
        frame = pd.DataFrame({"日期": dates})
        split = split_dates_train_val_test(
            frame,
            train_days=439,
            gap_days=5,
            val_days=39,
            gap2_days=24,
            test_days=5,
        )
        self.assertEqual(split["train"]["n_days"], 439)
        self.assertEqual(split["gap1"]["n_days"], 5)
        self.assertEqual(split["validation"]["n_days"], 39)
        self.assertEqual(split["gap2"]["n_days"], 24)
        self.assertEqual(split["test"]["n_days"], 5)
        self.assertLess(split["train"]["end"], split["gap1"]["start"])
        self.assertLess(split["gap1"]["end"], split["validation"]["start"])
        self.assertLess(split["validation"]["end"], split["gap2"]["start"])
        self.assertLess(split["gap2"]["end"], split["test"]["start"])

    def test_gap2_defaults_to_gap1(self):
        frame = pd.DataFrame({"日期": pd.bdate_range("2024-01-01", periods=30)})
        split = split_dates_train_val_test(
            frame, train_days=10, gap_days=3, val_days=5, test_days=4
        )
        self.assertEqual(split["gap2"]["n_days"], 3)

    def test_negative_gap2_is_rejected(self):
        frame = pd.DataFrame({"日期": pd.bdate_range("2024-01-01", periods=30)})
        with self.assertRaisesRegex(ValueError, "gap2_days"):
            split_dates_train_val_test(
                frame, train_days=10, gap_days=3, val_days=5,
                gap2_days=-1, test_days=4,
            )


if __name__ == "__main__":
    unittest.main()
