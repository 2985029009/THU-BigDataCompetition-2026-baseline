import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from pit_data import (  # noqa: E402
    build_membership,
    build_model_data,
    causal_adjust_prices,
    normalize_ts_code,
    point_in_time_cross_sectional_rank,
)
from utils import create_ranking_dataset_vectorized  # noqa: E402


class PointInTimeDataTests(unittest.TestCase):
    def test_numeric_excel_codes_are_normalized_without_decimal_suffix(self):
        self.assertEqual(normalize_ts_code(600000.0), "600000.SH")
        self.assertEqual(normalize_ts_code("1.0"), "000001.SZ")

    def test_causal_adjustment_is_invariant_to_future_append(self):
        base = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"] * 3,
                "trade_date": pd.date_range("2024-01-01", periods=3),
                "open": [10.0, 10.5, 5.5],
                "close": [10.0, 11.0, 5.5],
                "high": [10.2, 11.2, 5.6],
                "low": [9.8, 10.4, 5.4],
                "adj_factor": [1.0, 1.0, 2.0],
            }
        )
        future = pd.concat(
            [
                base,
                pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ"],
                        "trade_date": [pd.Timestamp("2024-01-04")],
                        "open": [3.0],
                        "close": [3.0],
                        "high": [3.1],
                        "low": [2.9],
                        "adj_factor": [4.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        left = causal_adjust_prices(base)
        right = causal_adjust_prices(future).iloc[: len(base)]
        pd.testing.assert_frame_equal(
            left[["open", "close", "high", "low"]],
            right[["open", "close", "high", "low"]],
        )

    def test_membership_changes_apply_on_effective_date(self):
        days = pd.bdate_range("2024-01-01", periods=4)
        snapshots = pd.DataFrame(
            {
                "trade_date": [days[0]] * 3 + [days[2]] * 3,
                "con_code": [
                    "000001.SZ",
                    "000002.SZ",
                    "600000.SH",
                    "000001.SZ",
                    "000003.SZ",
                    "600000.SH",
                ],
                "weight": [1.0] * 6,
            }
        )
        events = pd.DataFrame(
            {
                "effective_date": [days[1], days[1]],
                "con_code": ["000002.SZ", "000003.SZ"],
                "action": ["remove", "add"],
                "source_announcement": ["official-a", "official-a"],
            }
        )
        intervals, daily, audit = build_membership(
            snapshots, events, days, days[0], days[-1], expected_count=3
        )
        self.assertEqual(audit["status"], "passed")
        day0 = set(daily.loc[daily["日期"].eq(days[0]), "股票代码"])
        day1 = set(daily.loc[daily["日期"].eq(days[1]), "股票代码"])
        self.assertIn("000002", day0)
        self.assertNotIn("000002", day1)
        self.assertIn("000003", day1)
        self.assertEqual(daily.groupby("日期").size().tolist(), [3, 3, 3, 3])
        self.assertGreaterEqual(len(intervals), 4)

    def test_global_calendar_label_and_suspended_entry(self):
        days = pd.bdate_range("2024-01-01", periods=7)
        rows = []
        for code in ["000001.SZ", "000002.SZ"]:
            for index, day in enumerate(days):
                if code == "000001.SZ" and index == 1:
                    continue
                price = 10.0 + index
                rows.append(
                    {
                        "ts_code": code,
                        "trade_date": day,
                        "open": price,
                        "close": price + 0.5,
                        "high": price + 1.0,
                        "low": price - 1.0,
                        "vol": 100.0,
                        "amount": 10.0,
                        "turnover_rate": 1.0,
                        "adj_factor": 1.0,
                    }
                )
        market = pd.DataFrame(rows)
        membership = pd.DataFrame(
            [
                {"日期": day, "股票代码": code, "ts_code": f"{code}.SZ"}
                for day in days[:3]
                for code in ["000001", "000002"]
            ]
        )
        model = build_model_data(market, membership, days, days[0], days[2])
        failed = model[
            model["股票代码"].eq("000001") & model["日期"].eq(days[0])
        ].iloc[0]
        normal = model[
            model["股票代码"].eq("000002") & model["日期"].eq(days[0])
        ].iloc[0]
        removed = model[
            model["股票代码"].eq("000002") & model["日期"].eq(days[3])
        ].iloc[0]
        self.assertTrue(failed["label_entry_failed"])
        self.assertEqual(failed["label"], 0.0)
        self.assertAlmostEqual(normal["label"], (15.5 - 11.0) / 11.0)
        self.assertFalse(removed["is_member"])

    def test_cross_sectional_rank_uses_only_true_candidates_as_reference(self):
        frame = pd.DataFrame(
            {
                "日期": [pd.Timestamp("2024-01-01")] * 3,
                "x": [10.0, 20.0, 1000.0],
                "is_member": [True, True, False],
                "is_tradable": [True, True, True],
            }
        )
        ranked = point_in_time_cross_sectional_rank(frame, ["x"])
        self.assertEqual(ranked.loc[0, "x"], 0.5)
        self.assertEqual(ranked.loc[1, "x"], 1.0)
        self.assertEqual(ranked.loc[2, "x"], 1.0)

    def test_ranking_windows_only_end_on_pit_candidates(self):
        dates = pd.bdate_range("2024-01-01", periods=4)
        frame = pd.DataFrame(
            {
                "日期": list(dates) * 2,
                "instrument": [0] * 4 + [1] * 4,
                "feature": np.arange(8, dtype=float),
                "label": [0.1] * 8,
                "is_member": [True] * 4 + [False, False, True, True],
                "is_tradable": [True] * 8,
            }
        )
        seqs, _, _, stocks, sample_dates = create_ranking_dataset_vectorized(
            frame,
            ["feature"],
            sequence_length=2,
            return_dates=True,
            min_stocks=1,
        )
        by_date = {date: ids for date, ids in zip(sample_dates, stocks)}
        self.assertNotIn(1, by_date[pd.Timestamp(dates[1])])
        self.assertIn(1, by_date[pd.Timestamp(dates[2])])
        self.assertEqual(len(seqs), 3)


if __name__ == "__main__":
    unittest.main()
