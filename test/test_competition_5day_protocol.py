import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))

from train import split_dates_competition_5day  # noqa: E402


def test_competition_split_is_one_signal_plus_five_score_days():
    dates = pd.bdate_range("2025-01-01", periods=30)
    frame = pd.DataFrame({"日期": dates})
    split = split_dates_competition_5day(frame)
    assert split["validation"]["n_days"] == 0
    assert split["train"]["n_days"] == 20
    assert split["embargo_context"]["n_days"] == 5
    assert split["signal"]["date"] == dates[-6].strftime("%Y-%m-%d")
    assert split["test"]["start"] == dates[-5].strftime("%Y-%m-%d")
    assert split["test"]["end"] == dates[-1].strftime("%Y-%m-%d")
    assert split["test"]["n_days"] == 5


def test_last_training_label_can_end_on_signal_but_not_score_window():
    dates = pd.bdate_range("2025-01-01", periods=30)
    split = split_dates_competition_5day(pd.DataFrame({"日期": dates}))
    train_end = pd.Timestamp(split["train"]["end"])
    signal = pd.Timestamp(split["signal"]["date"])
    score_start = pd.Timestamp(split["test"]["start"])
    position = {date: i for i, date in enumerate(dates)}
    assert position[signal] - position[train_end] == 5
    assert signal < score_start


def test_fixed_400_day_rolling_window_has_410_days_total():
    dates = pd.bdate_range("2023-01-02", periods=500)
    split = split_dates_competition_5day(
        pd.DataFrame({"日期": dates}), train_days=400
    )
    assert split["window_trading_days"] == 410
    assert split["train"]["n_days"] == 400
    assert split["train"]["start"] == dates[-410].strftime("%Y-%m-%d")
    assert split["embargo_context"]["n_days"] == 5
    assert split["signal"]["date"] == dates[-6].strftime("%Y-%m-%d")
    assert split["test"]["n_days"] == 5


def test_explicit_training_cutoff_can_leave_extra_embargo_context():
    dates = pd.bdate_range("2026-06-01", "2026-07-17")
    frame = pd.DataFrame({"日期": dates})
    split = split_dates_competition_5day(
        frame, train_end_date="2026-07-01"
    )
    assert split["train"]["end"] == "2026-07-01"
    assert split["embargo_context"]["start"] == "2026-07-02"
    assert split["embargo_context"]["end"] == "2026-07-10"
    assert split["embargo_context"]["n_days"] == 7
    assert split["signal"]["date"] == "2026-07-10"
    assert split["test"]["start"] == "2026-07-13"
    assert split["test"]["end"] == "2026-07-17"
