"""Causal market-regime labels and auditable date-level stratified sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


REGIMES = ("weak", "normal", "strong")


def build_causal_market_state_table(data: pd.DataFrame) -> pd.DataFrame:
    """Build daily indicators using only information available through each date."""
    required = {"日期", "涨跌幅"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"市场状态缺少字段: {sorted(missing)}")
    daily = (
        data.assign(
            日期=pd.to_datetime(data["日期"]).dt.normalize(),
            _ret=pd.to_numeric(data["涨跌幅"], errors="coerce") / 100.0,
        )
        .groupby("日期", as_index=False)
        .agg(
            market_equal_weight_return=("_ret", "mean"),
            breadth_positive=("_ret", lambda x: float((x > 0).mean())),
            stocks_observed=("_ret", "count"),
        )
        .sort_values("日期")
        .reset_index(drop=True)
    )
    returns = daily["market_equal_weight_return"]
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    daily["market_return_20"] = (1.0 + returns).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1.0
    daily["market_volatility_20"] = returns.rolling(20, min_periods=20).std(ddof=0)
    daily["market_drawdown_60"] = wealth / wealth.rolling(60, min_periods=60).max() - 1.0
    daily["breadth_20"] = daily["breadth_positive"].rolling(20, min_periods=20).mean()
    daily["causal_feature_audit"] = "all indicators use rows dated <= signal_date"
    return daily


def fit_market_state_thresholds(state_table: pd.DataFrame, train_dates) -> dict:
    """Freeze predetermined 30/70-percentile thresholds on training dates only."""
    train_dates = pd.DatetimeIndex(pd.to_datetime(list(train_dates))).normalize()
    fit = state_table[state_table["日期"].isin(train_dates)].dropna(
        subset=["market_return_20", "market_volatility_20", "market_drawdown_60", "breadth_20"]
    )
    if len(fit) < 30:
        raise ValueError(f"可用于拟合市场状态阈值的训练日过少: {len(fit)}")
    return {
        "fit_start": fit["日期"].min().strftime("%Y-%m-%d"),
        "fit_end": fit["日期"].max().strftime("%Y-%m-%d"),
        "fit_days": int(len(fit)),
        "quantiles_frozen_before_training": [0.30, 0.70],
        "return20_q30": float(fit["market_return_20"].quantile(0.30)),
        "return20_q70": float(fit["market_return_20"].quantile(0.70)),
        "volatility20_q30": float(fit["market_volatility_20"].quantile(0.30)),
        "volatility20_q70": float(fit["market_volatility_20"].quantile(0.70)),
        "drawdown60_q30": float(fit["market_drawdown_60"].quantile(0.30)),
        "drawdown60_q70": float(fit["market_drawdown_60"].quantile(0.70)),
        "breadth20_q30": float(fit["breadth_20"].quantile(0.30)),
        "breadth20_q70": float(fit["breadth_20"].quantile(0.70)),
        "classification_rule": (
            "weak: >=3 of low return/high volatility/deep drawdown/low breadth; "
            "strong: not weak and >=3 of high return/low volatility/shallow drawdown/high breadth; "
            "otherwise normal"
        ),
    }


def apply_market_state_labels(state_table: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    result = state_table.copy()
    result["weak_low_return"] = result["market_return_20"] <= thresholds["return20_q30"]
    result["weak_high_volatility"] = result["market_volatility_20"] >= thresholds["volatility20_q70"]
    result["weak_deep_drawdown"] = result["market_drawdown_60"] <= thresholds["drawdown60_q30"]
    result["weak_low_breadth"] = result["breadth_20"] <= thresholds["breadth20_q30"]
    result["strong_high_return"] = result["market_return_20"] >= thresholds["return20_q70"]
    result["strong_low_volatility"] = result["market_volatility_20"] <= thresholds["volatility20_q30"]
    result["strong_shallow_drawdown"] = result["market_drawdown_60"] >= thresholds["drawdown60_q70"]
    result["strong_high_breadth"] = result["breadth_20"] >= thresholds["breadth20_q70"]
    weak_components = [
        "weak_low_return", "weak_high_volatility",
        "weak_deep_drawdown", "weak_low_breadth",
    ]
    strong_components = [
        "strong_high_return", "strong_low_volatility",
        "strong_shallow_drawdown", "strong_high_breadth",
    ]
    weak_score = result[weak_components].sum(axis=1)
    strong_score = result[strong_components].sum(axis=1)
    result["weak_score"] = weak_score
    result["strong_score"] = strong_score
    result["market_state"] = np.where(
        weak_score >= 3, "weak", np.where(strong_score >= 3, "strong", "normal")
    )
    return result


class AuditedStratifiedDateSampler(Sampler[int]):
    """Sample whole date-level dataset items at an exact weak target share."""

    def __init__(self, sample_dates, states, weak_target_share, seed, audit_file):
        self.sample_dates = pd.DatetimeIndex(pd.to_datetime(sample_dates)).normalize()
        self.states = np.asarray(list(states), dtype=object)
        if len(self.sample_dates) != len(self.states):
            raise ValueError("sample_dates与states长度不一致")
        unknown = set(self.states).difference(REGIMES)
        if unknown:
            raise ValueError(f"存在未知市场状态: {sorted(unknown)}")
        if not 0.0 < weak_target_share < 1.0:
            raise ValueError("weak_target_share必须在(0, 1)内")
        if not np.any(self.states == "weak"):
            raise ValueError("训练日期中没有weak状态，无法分层采样")
        self.weak_target_share = float(weak_target_share)
        self.seed = int(seed)
        self.audit_file = Path(audit_file)
        self.epoch = 0

    def __len__(self):
        return len(self.states)

    @staticmethod
    def _draw(indices, count, rng):
        indices = np.asarray(indices, dtype=int)
        if count <= len(indices):
            return rng.permutation(indices)[:count]
        extra = rng.choice(indices, size=count - len(indices), replace=True)
        return np.concatenate([rng.permutation(indices), extra])

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        total = len(self)
        targets = {"weak": int(round(total * self.weak_target_share))}
        remaining = total - targets["weak"]
        normal_n = int(np.sum(self.states == "normal"))
        strong_n = int(np.sum(self.states == "strong"))
        if normal_n + strong_n == 0:
            raise ValueError("训练日期中缺少normal/strong状态")
        targets["normal"] = int(round(remaining * normal_n / (normal_n + strong_n)))
        targets["strong"] = remaining - targets["normal"]

        pieces = []
        for state in REGIMES:
            indices = np.flatnonzero(self.states == state)
            if targets[state] and len(indices) == 0:
                raise ValueError(f"训练日期中没有{state}状态")
            pieces.append(self._draw(indices, targets[state], rng))
        sampled = np.concatenate(pieces)
        rng.shuffle(sampled)
        sampled_dates = self.sample_dates[sampled]
        payload = {
            "epoch": self.epoch + 1,
            "sampler_seed": self.seed,
            "epoch_seed": self.seed + self.epoch,
            "weak_target_share": self.weak_target_share,
            "sample_count": int(total),
            "state_counts": targets,
            "repeat_draws": int(total - pd.Index(sampled_dates).nunique()),
            "sampled_dates": [d.strftime("%Y-%m-%d") for d in sampled_dates],
        }
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.epoch += 1
        return iter(sampled.tolist())
