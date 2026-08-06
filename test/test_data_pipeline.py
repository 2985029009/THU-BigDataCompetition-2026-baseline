import os
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'code', 'src')
sys.path.insert(0, SRC)

from train import (
    ratio_split_day_counts,
    restrict_split_source_to_recent_years,
    split_dates_train_val_test,
)
from utils import create_ranking_dataset_vectorized


class ChronologicalSplitTests(unittest.TestCase):
    def test_recent_three_year_window_and_ratio_counts(self):
        dates = pd.bdate_range('2022-07-25', '2026-07-24')
        df = pd.DataFrame({'日期': dates, '股票代码': ['000001'] * len(dates)})

        recent, window = restrict_split_source_to_recent_years(df, 3)
        self.assertEqual(window['cutoff_date'], '2023-07-24')
        self.assertEqual(window['first_trading_date'], '2023-07-24')
        self.assertEqual(window['latest_trading_date'], '2026-07-24')

        train_days, val_days, test_days = ratio_split_day_counts(
            recent, ratios=(0.70, 0.15, 0.15), gap_days=5
        )
        self.assertEqual(
            train_days + val_days + test_days + 10,
            window['n_trading_days'],
        )
        usable = window['n_trading_days'] - 10
        self.assertAlmostEqual(train_days / usable, 0.70, places=2)
        self.assertAlmostEqual(val_days / usable, 0.15, places=2)
        self.assertAlmostEqual(test_days / usable, 0.15, places=2)

    def test_explicit_split_start_keeps_expected_window_boundaries(self):
        dates = pd.bdate_range('2024-07-17', periods=405)
        df = pd.DataFrame({'日期': dates, '股票代码': ['000001'] * len(dates)})
        split = split_dates_train_val_test(
            df, train_days=360, gap_days=5, val_days=40, test_days=0
        )
        self.assertEqual(split['train']['start'], dates[0].strftime('%Y-%m-%d'))
        self.assertEqual(split['train']['end'], dates[359].strftime('%Y-%m-%d'))
        self.assertEqual(
            split['validation']['start'], dates[365].strftime('%Y-%m-%d')
        )
        self.assertEqual(
            split['validation']['end'], dates[404].strftime('%Y-%m-%d')
        )

    def test_split_has_trading_day_purge_on_both_boundaries(self):
        dates = pd.bdate_range('2025-01-02', periods=180)
        df = pd.DataFrame({'日期': dates, '股票代码': ['000001'] * len(dates)})

        split = split_dates_train_val_test(
            df, train_days=60, gap_days=5, val_days=50, test_days=60
        )
        positions = {date: idx for idx, date in enumerate(dates)}

        self.assertEqual(
            positions[pd.Timestamp(split['validation']['start'])]
            - positions[pd.Timestamp(split['train']['end'])] - 1,
            5,
        )
        self.assertEqual(
            positions[pd.Timestamp(split['test']['start'])]
            - positions[pd.Timestamp(split['validation']['end'])] - 1,
            5,
        )
        self.assertLess(split['train']['end'], split['validation']['start'])
        self.assertLess(split['validation']['end'], split['test']['start'])


class RankingWindowTests(unittest.TestCase):
    def test_business_day_gaps_do_not_remove_valid_windows(self):
        dates = pd.bdate_range('2026-01-05', periods=10)
        rows = []
        for stock in range(10):
            for day_idx, date in enumerate(dates):
                rows.append({
                    'instrument': stock,
                    '日期': date,
                    'feature': float(stock + day_idx),
                    'label': float(stock - day_idx) / 100.0,
                })
        data = pd.DataFrame(rows)

        sequences, targets, relevance, stocks, sample_dates = (
            create_ranking_dataset_vectorized(
                data,
                ['feature'],
                sequence_length=3,
                return_dates=True,
            )
        )

        self.assertEqual(len(sequences), 8)
        self.assertEqual(list(sample_dates), list(dates[2:]))
        self.assertTrue(all(seq.shape == (10, 3, 1) for seq in sequences))
        self.assertTrue(all(target.shape == (10,) for target in targets))
        self.assertTrue(all(score.shape == (10,) for score in relevance))
        self.assertTrue(all(len(day_stocks) == 10 for day_stocks in stocks))


if __name__ == '__main__':
    unittest.main()
