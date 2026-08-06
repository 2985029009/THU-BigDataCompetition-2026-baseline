import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from config import config
from utils import (
    BASELINE24_FEATURES,
    COMPONENT2_FEATURES,
    CROSS4_FEATURES,
    MARKET4_FEATURES,
    engineer_features_baseline24,
)


def _sample_stock_frame(n_rows=120):
    index = np.arange(n_rows, dtype=float)
    close = 10.0 + index * 0.02 + np.sin(index / 7.0) * 0.2
    return pd.DataFrame(
        {
            "股票代码": ["000001"] * n_rows,
            "日期": pd.date_range("2025-01-01", periods=n_rows, freq="B"),
            "开盘": close * 0.998,
            "收盘": close,
            "最高": close * 1.01,
            "最低": close * 0.99,
            "成交量": 1_000_000.0 + index * 1_000.0,
            "成交额": (1_000_000.0 + index * 1_000.0) * close,
            "振幅": np.full(n_rows, 0.02),
            "涨跌额": pd.Series(close).diff().fillna(0.0),
            "换手率": 0.01 + index * 0.00001,
            "涨跌幅": pd.Series(close).pct_change().fillna(0.0),
        }
    )


def test_baseline24_entry_only_builds_fixed_features():
    frame = _sample_stock_frame()
    native = engineer_features_baseline24(frame)

    assert config["feature_num"] == "baseline24_market4"
    assert config["include_features"] == BASELINE24_FEATURES + MARKET4_FEATURES
    assert np.isfinite(native[BASELINE24_FEATURES].to_numpy(dtype=float)).all()
    assert {"KMID", "BETA20", "CORR20"}.isdisjoint(native.columns)
