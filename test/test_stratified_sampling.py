import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from stratified_sampling import (  # noqa: E402
    AuditedStratifiedDateSampler,
    apply_market_state_labels,
    build_causal_market_state_table,
    fit_market_state_thresholds,
)


def test_market_state_features_are_causal():
    dates = pd.bdate_range("2024-01-01", periods=100)
    rows = [
        {"日期": date, "股票代码": stock, "涨跌幅": np.sin(i / 7) + stock / 100}
        for i, date in enumerate(dates)
        for stock in range(20)
    ]
    data = pd.DataFrame(rows)
    original = build_causal_market_state_table(data)
    changed = data.copy()
    changed.loc[changed["日期"] > dates[70], "涨跌幅"] = 99.0
    perturbed = build_causal_market_state_table(changed)
    columns = ["market_return_20", "market_volatility_20", "market_drawdown_60", "breadth_20"]
    pd.testing.assert_frame_equal(
        original.loc[:70, columns], perturbed.loc[:70, columns], check_exact=True
    )


def test_thresholds_and_labels_have_three_valid_states():
    dates = pd.bdate_range("2024-01-01", periods=180)
    market = np.r_[np.full(60, -1.0), np.zeros(60), np.full(60, 1.0)]
    data = pd.DataFrame(
        [
            {"日期": date, "股票代码": stock, "涨跌幅": market[i] + stock * 0.001}
            for i, date in enumerate(dates)
            for stock in range(20)
        ]
    )
    table = build_causal_market_state_table(data)
    thresholds = fit_market_state_thresholds(table, dates)
    labelled = apply_market_state_labels(table, thresholds)
    assert set(labelled["market_state"]).issubset({"weak", "normal", "strong"})
    assert {"weak", "strong"}.issubset(set(labelled["market_state"]))


def test_sampler_hits_exact_weak_share_and_audits_dates(tmp_path):
    dates = pd.bdate_range("2025-01-01", periods=100)
    states = ["weak"] * 10 + ["normal"] * 60 + ["strong"] * 30
    audit = tmp_path / "sampling.jsonl"
    sampler = AuditedStratifiedDateSampler(dates, states, 0.30, 42, audit)
    sampled = list(iter(sampler))
    assert len(sampled) == 100
    assert sum(states[index] == "weak" for index in sampled) == 30
    payload = json.loads(audit.read_text(encoding="utf-8").splitlines()[0])
    assert payload["state_counts"]["weak"] == 30
    assert payload["repeat_draws"] >= 20
    assert len(payload["sampled_dates"]) == 100
