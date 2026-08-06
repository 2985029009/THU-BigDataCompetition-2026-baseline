"""Reversible, regime-gated post-processing for contest Top-5 selection.

This module is deliberately independent from model training.  It preserves the
raw model ranking, applies a frozen three-family overlay only when explicitly
enabled *and* the causal regime rule is true, and always emits equal-weight
Top-5 output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


FAMILY_SPECS = {
    "low_volatility": {
        "factors": [
            "振幅", "STD5", "STD10", "STD20", "STD60",
            "volatility_10", "volatility_20", "atr_14",
        ],
        "direction": "lower_is_better",
    },
    "low_activity": {
        "factors": ["成交额", "换手率"],
        "direction": "lower_is_better",
    },
    "long_term_reversal": {
        "factors": ["ROC60"],
        "direction": "higher_is_better",
    },
}

REGIME_THRESHOLDS = {
    "breadth_today_min": 0.50,
    "breadth_change_5_20_min": -0.01,
    "vol20_hist_percentile_min": 1.0 / 3.0,
}


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = False
    alpha: float = 0.0
    top_k: int = 5
    stock_id_column: str = "股票代码"
    model_score_column: str = "prediction_score"

    def validate(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha必须位于[0, 1]，当前为{self.alpha}")
        if self.top_k != 5:
            raise ValueError("比赛规则固定只允许Top-5，本层拒绝其他top_k")


@dataclass(frozen=True)
class PoolRerankConfig:
    enabled: bool = False
    pool_k: int = 15
    top_k: int = 5
    family_subset: tuple[str, ...] = ("low_volatility",)
    stock_id_column: str = "股票代码"
    model_score_column: str = "prediction_score"

    def validate(self) -> None:
        if self.pool_k != 15 or self.top_k != 5:
            raise ValueError("当前冻结协议只允许原模型Top-15池内重排并最终输出Top-5")
        if self.top_k > self.pool_k:
            raise ValueError("top_k不得大于pool_k")
        unknown = [name for name in self.family_subset if name not in FAMILY_SPECS]
        if not self.family_subset or unknown:
            raise ValueError(f"无效风格族子集: {self.family_subset}, unknown={unknown}")


def evaluate_frozen_regime_rule(metrics: Mapping[str, float]) -> dict[str, object]:
    """Evaluate the frozen causal rule from signal-date-or-earlier metrics."""
    required = ["breadth_today", "breadth_change_5_20", "vol20_hist_percentile"]
    missing = [name for name in required if name not in metrics]
    if missing:
        raise KeyError(f"Regime指标缺失: {missing}")
    values = {name: float(metrics[name]) for name in required}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Regime指标必须全部为有限数值")
    component_pass = {
        "breadth_today": values["breadth_today"] >= REGIME_THRESHOLDS["breadth_today_min"],
        "breadth_trend": values["breadth_change_5_20"] >= REGIME_THRESHOLDS["breadth_change_5_20_min"],
        "volatility": values["vol20_hist_percentile"] >= REGIME_THRESHOLDS["vol20_hist_percentile_min"],
    }
    margins = {
        "breadth_today": (values["breadth_today"] - 0.50) / 0.05,
        "breadth_trend": (values["breadth_change_5_20"] + 0.01) / 0.05,
        "volatility": (values["vol20_hist_percentile"] - 1.0 / 3.0) / 0.10,
    }
    return {
        "triggered": bool(all(component_pass.values())),
        "component_pass": component_pass,
        "component_margins": margins,
        "causal_trigger_margin": float(min(margins.values())),
        "input_metrics": values,
        "thresholds": REGIME_THRESHOLDS,
    }


def _percentile_rank(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"排序字段{series.name!r}包含缺失或非有限值")
    return numeric.rank(method="average", pct=True, ascending=higher_is_better)


def compute_family_overlay(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute factor-direction percentiles and equal-weight family composite."""
    required = sorted({factor for spec in FAMILY_SPECS.values() for factor in spec["factors"]})
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"重排所需因子缺失: {missing}")
    result = pd.DataFrame(index=frame.index)
    for family, spec in FAMILY_SPECS.items():
        directional = []
        for factor in spec["factors"]:
            higher_is_better = spec["direction"] == "higher_is_better"
            directional.append(
                _percentile_rank(frame[factor], higher_is_better=higher_is_better)
            )
        result[family] = pd.concat(directional, axis=1).mean(axis=1)
    result["reversal_style_composite"] = result[list(FAMILY_SPECS)].mean(axis=1)
    result["family_rank_pct"] = _percentile_rank(
        result["reversal_style_composite"], higher_is_better=True
    )
    return result


def rerank_candidates(
    frame: pd.DataFrame,
    *,
    regime_triggered: bool,
    config: RerankConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return full audit ranking and a compact rollback manifest."""
    config.validate()
    if config.stock_id_column not in frame.columns:
        raise KeyError(f"股票代码列不存在: {config.stock_id_column}")
    if config.model_score_column not in frame.columns:
        raise KeyError(f"模型分数列不存在: {config.model_score_column}")
    if len(frame) < config.top_k:
        raise ValueError(f"候选股票不足{config.top_k}只，当前为{len(frame)}只")
    if frame[config.stock_id_column].astype(str).duplicated().any():
        raise ValueError("候选股票代码必须唯一")

    audit = frame.copy()
    audit[config.stock_id_column] = audit[config.stock_id_column].astype(str).str.zfill(6)
    overlay = compute_family_overlay(audit)
    for column in overlay.columns:
        audit[column] = overlay[column]
    audit["model_rank_pct"] = _percentile_rank(
        audit[config.model_score_column], higher_is_better=True
    )

    effective_alpha = float(config.alpha if config.enabled and regime_triggered else 0.0)
    audit["configured_alpha"] = float(config.alpha)
    audit["effective_alpha"] = effective_alpha
    audit["regime_triggered"] = bool(regime_triggered)
    audit["rerank_enabled"] = bool(config.enabled)
    audit["final_score"] = (
        (1.0 - effective_alpha) * audit["model_rank_pct"]
        + effective_alpha * audit["family_rank_pct"]
    )

    # Match the existing predict.py contract exactly, including its np.argsort
    # tie behavior, so alpha=0/disabled is a byte-level Top-5 rollback path.
    model_values = pd.to_numeric(audit[config.model_score_column]).to_numpy()
    raw_positions = np.argsort(model_values)[::-1]
    raw_order = audit.iloc[raw_positions]
    raw_rank = pd.Series(
        np.arange(1, len(raw_order) + 1), index=raw_order.index, dtype=int
    )
    audit["original_rank"] = raw_rank.reindex(audit.index)

    if effective_alpha == 0.0:
        final_order = raw_order
    else:
        final_order = audit.sort_values(
            ["final_score", "model_rank_pct", "family_rank_pct", config.stock_id_column],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
    final_rank = pd.Series(
        np.arange(1, len(final_order) + 1), index=final_order.index, dtype=int
    )
    audit["final_rank"] = final_rank.reindex(audit.index)
    audit["rank_change"] = audit["original_rank"] - audit["final_rank"]
    audit["in_original_top5"] = audit["original_rank"] <= config.top_k
    audit["in_final_top5"] = audit["final_rank"] <= config.top_k
    audit = audit.sort_values("final_rank", kind="mergesort").reset_index(drop=True)

    original_top5 = (
        audit.sort_values("original_rank")[config.stock_id_column].head(config.top_k).tolist()
    )
    final_top5 = audit[config.stock_id_column].head(config.top_k).tolist()
    overlap = len(set(original_top5) & set(final_top5))
    manifest = {
        "schema_version": 1,
        "layer_name": "reversible_regime_rerank_v1",
        "enabled": bool(config.enabled),
        "regime_triggered": bool(regime_triggered),
        "configured_alpha": float(config.alpha),
        "effective_alpha": effective_alpha,
        "rollback_active": effective_alpha == 0.0,
        "candidate_count": len(audit),
        "top_k": config.top_k,
        "weights": [0.2] * config.top_k,
        "original_top5": original_top5,
        "final_top5": final_top5,
        "top5_overlap": overlap,
        "top5_changed_count": config.top_k - overlap,
        "family_specs": FAMILY_SPECS,
        "config": asdict(config),
    }
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    manifest["audit_sha256"] = hashlib.sha256(canonical).hexdigest()
    return audit, manifest


def rerank_within_model_pool(
    frame: pd.DataFrame,
    *,
    regime_triggered: bool,
    config: PoolRerankConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Rerank only the original model Top-15 and still emit Top-5.

    No continuous alpha exists in this protocol.  When disabled or untriggered,
    the final order is exactly the existing np.argsort model order.
    """
    config.validate()
    for column in (config.stock_id_column, config.model_score_column):
        if column not in frame.columns:
            raise KeyError(f"必要列不存在: {column}")
    if len(frame) < config.pool_k:
        raise ValueError(f"候选股票不足{config.pool_k}只，当前为{len(frame)}只")
    if frame[config.stock_id_column].astype(str).duplicated().any():
        raise ValueError("候选股票代码必须唯一")

    audit = frame.copy()
    audit[config.stock_id_column] = audit[config.stock_id_column].astype(str).str.zfill(6)
    overlay = compute_family_overlay(audit)
    for column in overlay.columns:
        audit[column] = overlay[column]
    audit["model_rank_pct"] = _percentile_rank(
        audit[config.model_score_column], higher_is_better=True
    )
    audit["pool_family_score"] = audit[list(config.family_subset)].mean(axis=1)
    audit["pool_family_rank_pct"] = _percentile_rank(
        audit["pool_family_score"], higher_is_better=True
    )

    model_values = pd.to_numeric(audit[config.model_score_column]).to_numpy()
    raw_positions = np.argsort(model_values)[::-1]
    raw_order = audit.iloc[raw_positions]
    audit["original_rank"] = pd.Series(
        np.arange(1, len(audit) + 1), index=raw_order.index, dtype=int
    ).reindex(audit.index)
    audit["in_original_top15"] = audit["original_rank"] <= config.pool_k
    active = bool(config.enabled and regime_triggered)

    if active:
        pool = audit[audit["in_original_top15"]].sort_values(
            ["pool_family_score", "model_rank_pct", config.stock_id_column],
            ascending=[False, False, True],
            kind="mergesort",
        )
        outside = audit[~audit["in_original_top15"]].sort_values(
            "original_rank", kind="mergesort"
        )
        final_order = pd.concat([pool, outside], axis=0)
    else:
        final_order = raw_order

    audit["final_rank"] = pd.Series(
        np.arange(1, len(audit) + 1), index=final_order.index, dtype=int
    ).reindex(audit.index)
    audit["rank_change"] = audit["original_rank"] - audit["final_rank"]
    audit["in_original_top5"] = audit["original_rank"] <= config.top_k
    audit["in_final_top5"] = audit["final_rank"] <= config.top_k
    audit["regime_triggered"] = bool(regime_triggered)
    audit["rerank_enabled"] = bool(config.enabled)
    audit["pool_rerank_active"] = active
    audit["configured_alpha"] = 0.0
    audit["effective_alpha"] = 0.0
    audit["final_score"] = -audit["final_rank"].astype(float)
    audit = audit.sort_values("final_rank", kind="mergesort").reset_index(drop=True)

    original_top5 = (
        audit.sort_values("original_rank")[config.stock_id_column].head(config.top_k).tolist()
    )
    original_top15 = (
        audit.sort_values("original_rank")[config.stock_id_column].head(config.pool_k).tolist()
    )
    final_top5 = audit[config.stock_id_column].head(config.top_k).tolist()
    if not set(final_top5).issubset(set(original_top15)):
        raise AssertionError("池内重排输出越过原模型Top-15边界")
    overlap = len(set(original_top5) & set(final_top5))
    manifest = {
        "schema_version": 1,
        "layer_name": "reversible_top15_internal_regime_rerank_v1",
        "enabled": bool(config.enabled),
        "regime_triggered": bool(regime_triggered),
        "pool_rerank_active": active,
        "rollback_active": not active,
        "candidate_count": len(audit),
        "pool_k": config.pool_k,
        "top_k": config.top_k,
        "weights": [0.2] * config.top_k,
        "family_subset": list(config.family_subset),
        "original_top5": original_top5,
        "original_top15": original_top15,
        "final_top5": final_top5,
        "final_top5_inside_original_top15": True,
        "top5_overlap": overlap,
        "top5_changed_count": config.top_k - overlap,
        "config": asdict(config),
    }
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    manifest["audit_sha256"] = hashlib.sha256(canonical).hexdigest()
    return audit, manifest


def write_rerank_outputs(
    audit: pd.DataFrame,
    manifest: Mapping[str, object],
    output_dir: str | Path,
    *,
    stock_id_column: str = "股票代码",
    overwrite: bool = False,
) -> Path:
    """Write original/final outputs and full audit trail atomically enough for review."""
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {target}")
    target.mkdir(parents=True, exist_ok=True)
    final_manifest = dict(manifest)
    final_manifest.pop("audit_sha256", None)
    canonical = json.dumps(
        final_manifest, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    final_manifest["audit_sha256"] = hashlib.sha256(canonical).hexdigest()
    top_k = int(final_manifest["top_k"])
    original = audit.sort_values("original_rank").head(top_k)
    final = audit.sort_values("final_rank").head(top_k)
    original[[stock_id_column]].rename(columns={stock_id_column: "stock_id"}).assign(
        weight=0.2
    ).to_csv(target / "original_result.csv", index=False, encoding="utf-8-sig")
    final[[stock_id_column]].rename(columns={stock_id_column: "stock_id"}).assign(
        weight=0.2
    ).to_csv(target / "result.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(target / "ranking_audit.csv", index=False, encoding="utf-8-sig")
    (target / "rerank_audit.json").write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target
