import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from pit_data import point_in_time_cross_sectional_rank  # noqa: E402
from utils import (  # noqa: E402
    CROSS4_FEATURES,
    COMPONENT2_FEATURES,
    LOW_VOLATILITY_FEATURES,
    MARKET4_FEATURES,
    add_causal_market_features,
    add_market_cross_features,
    add_market_component_cross_features,
    fit_market_stress_statistics,
)


def _market_panel(n_days=70):
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    rows = []
    for day_index, date in enumerate(dates):
        for stock_index, stock_code in enumerate(("000001", "000002")):
            rows.append(
                {
                    "股票代码": stock_code,
                    "日期": date,
                    "涨跌幅": 1.0 + stock_index,
                    "is_member": True,
                    "is_tradable": True,
                    "stock_feature": float(stock_index + day_index),
                }
            )
    return pd.DataFrame(rows), dates


def test_market4_are_causal_and_shared_within_each_date():
    panel, dates = _market_panel()
    featured = add_causal_market_features(panel)

    first_complete_20 = featured[featured["日期"].eq(dates[19])]
    assert first_complete_20["market_return_20"].nunique() == 1
    assert first_complete_20["market_volatility_20"].nunique() == 1
    np.testing.assert_allclose(
        first_complete_20["market_return_20"].iloc[0],
        (1.0 + 0.015) ** 20 - 1.0,
    )
    assert featured.loc[featured["日期"].lt(dates[19]), "market_return_20"].isna().all()
    assert featured.loc[featured["日期"].lt(dates[59]), "market_drawdown_60"].isna().all()

    changed_future = panel.copy()
    changed_future.loc[changed_future["日期"].gt(dates[40]), "涨跌幅"] = -50.0
    changed_featured = add_causal_market_features(changed_future)
    original_past = featured.loc[featured["日期"].le(dates[40]), MARKET4_FEATURES]
    changed_past = changed_featured.loc[
        changed_featured["日期"].le(dates[40]), MARKET4_FEATURES
    ]
    np.testing.assert_allclose(
        original_past.to_numpy(), changed_past.to_numpy(), equal_nan=True
    )


def test_market4_use_tradable_members_and_skip_cross_sectional_ranking():
    panel, dates = _market_panel()
    extreme = panel[panel["股票代码"].eq("000002")].index
    panel.loc[extreme, "涨跌幅"] = 1000.0
    panel.loc[extreme, "is_tradable"] = False
    featured = add_causal_market_features(panel)

    expected_return = (1.0 + 0.01) ** 20 - 1.0
    actual_return = featured.loc[
        featured["日期"].eq(dates[19]), "market_return_20"
    ].iloc[0]
    np.testing.assert_allclose(actual_return, expected_return)

    ranked = point_in_time_cross_sectional_rank(featured, ["stock_feature"])
    np.testing.assert_allclose(
        ranked[MARKET4_FEATURES].to_numpy(),
        featured[MARKET4_FEATURES].to_numpy(),
        equal_nan=True,
    )


def test_market_cross_features_follow_frozen_formula_and_training_statistics():
    rows = []
    dates = pd.bdate_range("2025-01-02", periods=4)
    market_values = [
        (-0.02, 0.01, -0.03, 0.45),
        (0.00, 0.02, -0.02, 0.50),
        (0.02, 0.03, -0.01, 0.55),
        (5.00, 5.00, 5.00, 5.00),
    ]
    for day_index, (date, market) in enumerate(zip(dates, market_values)):
        for stock_index in range(2):
            row = {
                "日期": date,
                **dict(zip(MARKET4_FEATURES, market)),
                "ROC60": 0.2 + 0.4 * stock_index,
                "RANK20": 0.3 + 0.2 * stock_index,
                "volume_change": 0.4 + 0.1 * stock_index,
            }
            row.update({name: 0.1 + 0.1 * stock_index for name in LOW_VOLATILITY_FEATURES})
            rows.append(row)
    frame = pd.DataFrame(rows)
    fit_rows = frame["日期"].le(dates[2])
    _, center, scale = fit_market_stress_statistics(frame, fit_rows)
    crossed = add_market_cross_features(frame, center, scale)

    # 未参与拟合的极端未来日不得改变训练统计量。
    np.testing.assert_allclose(center.to_numpy(), np.mean(market_values[:3], axis=0))
    expected_stress = np.mean(
        ((np.asarray(market_values[0]) - center.to_numpy()) / scale.to_numpy())
        * np.asarray([-1.0, 1.0, -1.0, -1.0])
    )
    first = crossed.iloc[0]
    np.testing.assert_allclose(first["stress_x_ROC60"], expected_stress * 0.2)
    np.testing.assert_allclose(first["stress_x_low_vol"], expected_stress * 0.9)
    assert np.isfinite(crossed[CROSS4_FEATURES].to_numpy()).all()


def test_market_component_crosses_use_separate_stress_dimensions():
    frame = pd.DataFrame({
        **{
            name: [0.0, 0.0]
            for name in MARKET4_FEATURES
        },
        "market_volatility_20": [3.0, -1.0],
        "market_drawdown_60": [-2.0, 2.0],
        "ROC20": [0.25, 0.75],
        "STD20": [0.40, 0.60],
    })
    crossed = add_market_component_cross_features(
        frame,
        market_center=[0.0, 1.0, 0.0, 0.0],
        market_scale=[1.0, 2.0, 2.0, 1.0],
    )
    np.testing.assert_allclose(
        crossed["volatility_stress_x_ROC20"], [0.25, -0.75]
    )
    np.testing.assert_allclose(
        crossed["drawdown_stress_x_STD20"], [0.40, -0.60]
    )
    assert np.isfinite(crossed[COMPONENT2_FEATURES].to_numpy()).all()
