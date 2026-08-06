"""Point-in-time universe and causal market-data utilities.

This module deliberately contains no network access.  The downloader in
``data/build_hs300_pit.py`` persists raw responses first and calls these pure
functions to build auditable outputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["开盘", "收盘", "最高", "最低"]
MODEL_COLUMNS = [
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
    "adj_factor",
    "is_member",
    "is_suspended",
    "is_tradable",
    "label",
    "label_entry_failed",
    "source",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_contest_score_label(
    frame: pd.DataFrame, output_column: str = "label"
) -> pd.DataFrame:
    """按score_self.py口径构造标签：未来第1条记录开盘到第5条记录开盘。"""
    required = {"股票代码", "日期", "开盘"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"比赛标签输入缺少字段: {sorted(missing)}")

    result = frame.copy()
    result["日期"] = pd.to_datetime(result["日期"])
    result = result.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    opens = pd.to_numeric(result["开盘"], errors="coerce")
    grouped_opens = opens.groupby(result["股票代码"], sort=False)
    entry_open = grouped_opens.shift(-1)
    exit_open = grouped_opens.shift(-5)
    result[output_column] = np.where(
        entry_open.notna() & entry_open.gt(0),
        (exit_open - entry_open) / entry_open,
        np.nan,
    )
    return result


def normalize_ts_code(value: object) -> str:
    text = str(value).strip().upper()
    numeric_match = pd.notna(value) and re.fullmatch(r"\d{1,6}(?:\.0+)?", text)
    if numeric_match:
        digits = f"{int(float(text)):06d}"
        exchange = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
        return f"{digits}.{exchange}"
    if re.fullmatch(r"\d{6}\.(?:SH|SZ)", text):
        digits, exchange = text.split(".", 1)
        return f"{digits.zfill(6)}.{exchange}"
    digits = "".join(char for char in text if char.isdigit()).zfill(6)
    exchange = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
    return f"{digits}.{exchange}"


def pure_code(value: object) -> str:
    return normalize_ts_code(value).split(".", 1)[0]


def causal_adjust_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply an expanding, endpoint-invariant adjustment factor.

    ``adj_factor`` is cumulative as of each row.  Normalising by the first
    factor is a scale choice based only on information available at the start;
    appending future rows therefore cannot change earlier adjusted prices.
    """
    required = {"ts_code", "trade_date", "open", "close", "high", "low", "adj_factor"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"因果复权输入缺少字段: {sorted(missing)}")

    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    numeric = ["open", "close", "high", "low", "adj_factor"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    if result[numeric].isna().any().any():
        raise ValueError("因果复权输入存在无法解析的价格或复权因子")
    if result["adj_factor"].le(0).any():
        raise ValueError("adj_factor 必须为正数")

    base_factor = result.groupby("ts_code", sort=False)["adj_factor"].transform("first")
    scale = result["adj_factor"] / base_factor
    for column in ["open", "close", "high", "low"]:
        result[column] = result[column] * scale
    return result


def build_membership(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    trading_days: Iterable[object],
    start_date: object,
    end_date: object,
    expected_count: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Replay official add/remove events from a Tushare snapshot anchor."""
    required_snapshot = {"trade_date", "con_code"}
    required_event = {"effective_date", "con_code", "action", "source_announcement"}
    if missing := required_snapshot.difference(snapshots.columns):
        raise ValueError(f"成分快照缺少字段: {sorted(missing)}")
    if missing := required_event.difference(events.columns):
        raise ValueError(f"调整事件缺少字段: {sorted(missing)}")

    snapshots = snapshots.copy()
    snapshots["trade_date"] = pd.to_datetime(snapshots["trade_date"]).dt.normalize()
    snapshots["con_code"] = snapshots["con_code"].map(normalize_ts_code)
    if "weight" not in snapshots:
        snapshots["weight"] = np.nan
    snapshots = snapshots.drop_duplicates(["trade_date", "con_code"], keep="last")

    events = events.copy()
    events["effective_date"] = pd.to_datetime(events["effective_date"]).dt.normalize()
    events["con_code"] = events["con_code"].map(normalize_ts_code)
    events["action"] = events["action"].str.lower()
    invalid_actions = set(events["action"]).difference({"add", "remove"})
    if invalid_actions:
        raise ValueError(f"未知成分调整动作: {sorted(invalid_actions)}")

    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_days))).normalize().unique().sort_values()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    calendar = calendar[(calendar >= start) & (calendar <= end)]
    if calendar.empty:
        raise ValueError("研究区间内没有交易日")

    anchors = snapshots[snapshots["trade_date"] <= calendar[0]]
    if anchors.empty:
        raise ValueError("开始日前没有可用的 Tushare 成分快照，禁止使用未来快照反推")
    anchor_date = anchors["trade_date"].max()
    anchor_rows = snapshots[snapshots["trade_date"].eq(anchor_date)]
    state = set(anchor_rows["con_code"])
    if len(state) != expected_count:
        raise ValueError(f"锚点 {anchor_date.date()} 成分数为 {len(state)}，预期 {expected_count}")

    weights = dict(zip(anchor_rows["con_code"], anchor_rows["weight"]))
    weight_source_date = anchor_date
    # The anchor snapshot already reflects all events effective on or before it.
    events = events[events["effective_date"] > anchor_date].copy()
    events_by_date = {
        date: group.sort_values(["action", "con_code"])
        for date, group in events.groupby("effective_date", sort=True)
    }
    snapshots_by_date = {
        date: group
        for date, group in snapshots.groupby("trade_date", sort=True)
        if date >= anchor_date
    }

    daily_rows: list[dict] = []
    mismatches: list[dict] = []
    state_by_day: dict[pd.Timestamp, set[str]] = {}
    event_sources: dict[tuple[str, str], str] = {}

    replay_days = pd.DatetimeIndex(
        sorted(set(calendar.tolist()) | set(snapshots_by_date) | set(events_by_date))
    )
    replay_days = replay_days[(replay_days >= anchor_date) & (replay_days <= calendar[-1])]
    for day in replay_days:
        if day in events_by_date:
            for row in events_by_date[day].itertuples(index=False):
                if row.action == "remove":
                    state.discard(row.con_code)
                else:
                    state.add(row.con_code)
                    weights.setdefault(row.con_code, np.nan)
                event_sources[(row.con_code, row.action)] = row.source_announcement

        if day in snapshots_by_date:
            snap = snapshots_by_date[day]
            expected = set(snap["con_code"])
            if state != expected:
                mismatches.append(
                    {
                        "trade_date": day.strftime("%Y-%m-%d"),
                        "missing_from_rebuild": sorted(expected - state),
                        "extra_in_rebuild": sorted(state - expected),
                    }
                )
            else:
                weights = dict(zip(snap["con_code"], snap["weight"]))
                weight_source_date = day

        if day in calendar:
            state_by_day[day] = set(state)
            for code in sorted(state):
                daily_rows.append(
                    {
                        "日期": day,
                        "股票代码": pure_code(code),
                        "ts_code": code,
                        "是否成分": 1,
                        "权重": weights.get(code, np.nan),
                        "来源快照日期": weight_source_date,
                    }
                )

    daily = pd.DataFrame(daily_rows)
    daily_counts = daily.groupby("日期").size()
    bad_counts = daily_counts[daily_counts.ne(expected_count)]
    if not bad_counts.empty:
        mismatches.append(
            {
                "type": "daily_count",
                "dates": {str(key.date()): int(value) for key, value in bad_counts.items()},
            }
        )

    union_codes = sorted(set().union(*state_by_day.values()))
    interval_rows: list[dict] = []
    for code in union_codes:
        active_days = [day for day in calendar if code in state_by_day[day]]
        if not active_days:
            continue
        block_start = active_days[0]
        previous = active_days[0]
        for day in active_days[1:] + [None]:
            contiguous = day is not None and calendar.get_loc(day) == calendar.get_loc(previous) + 1
            if contiguous:
                previous = day
                continue
            add_source = event_sources.get((code, "add"), f"tushare_snapshot:{anchor_date.date()}")
            remove_source = event_sources.get((code, "remove"), "")
            interval_rows.append(
                {
                    "股票代码": pure_code(code),
                    "纳入日期": block_start,
                    "调出日期": previous,
                    "来源公告": add_source or remove_source,
                    "校验状态": "通过" if not mismatches else "失败",
                }
            )
            if day is not None:
                block_start = day
                previous = day

    intervals = pd.DataFrame(interval_rows)
    audit = {
        "anchor_date": anchor_date.strftime("%Y-%m-%d"),
        "calendar_days": int(len(calendar)),
        "union_stock_count": int(len(union_codes)),
        "snapshot_dates": int(snapshots["trade_date"].nunique()),
        "event_count": int(len(events)),
        "mismatches": mismatches,
        "status": "passed" if not mismatches else "failed",
    }
    return intervals, daily, audit


def build_model_data(
    market: pd.DataFrame,
    daily_membership: pd.DataFrame,
    trading_days: Iterable[object],
    signal_start: object,
    signal_end: object,
) -> pd.DataFrame:
    """Build a global-calendar panel and executable t+1-open/t+5-close labels."""
    required = {
        "ts_code",
        "trade_date",
        "open",
        "close",
        "high",
        "low",
        "vol",
        "amount",
        "turnover_rate",
        "adj_factor",
    }
    if missing := required.difference(market.columns):
        raise ValueError(f"行情数据缺少字段: {sorted(missing)}")

    adjusted = causal_adjust_prices(market)
    adjusted["ts_code"] = adjusted["ts_code"].map(normalize_ts_code)
    adjusted["股票代码"] = adjusted["ts_code"].map(pure_code)
    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_days))).normalize().unique().sort_values()
    calendar_position = {day: index for index, day in enumerate(calendar)}

    membership = daily_membership.copy()
    membership["日期"] = pd.to_datetime(membership["日期"]).dt.normalize()
    membership["股票代码"] = membership["股票代码"].astype(str).str.zfill(6)
    member_keys = set(zip(membership["股票代码"], membership["日期"]))

    panel_rows: list[pd.DataFrame] = []
    for ts_code, stock in adjusted.groupby("ts_code", sort=True):
        stock = stock.sort_values("trade_date").set_index("trade_date")
        listed_calendar = calendar[
            (calendar >= stock.index.min()) & (calendar <= calendar[-1])
        ]
        panel = stock.reindex(listed_calendar)
        panel.index.name = "日期"
        panel["ts_code"] = ts_code
        panel["股票代码"] = pure_code(ts_code)
        panel["is_suspended"] = panel["close"].isna()

        previous_close = panel["close"].ffill()
        for column in ["open", "close", "high", "low"]:
            panel[column] = panel[column].fillna(previous_close)
        panel["adj_factor"] = panel["adj_factor"].ffill()
        panel[["vol", "amount", "turnover_rate"]] = panel[
            ["vol", "amount", "turnover_rate"]
        ].fillna(0.0)
        panel["is_member"] = [
            (pure_code(ts_code), day) in member_keys for day in panel.index
        ]
        panel["is_tradable"] = (~panel["is_suspended"]) & panel["is_member"]
        panel_rows.append(panel.reset_index())

    panel = pd.concat(panel_rows, ignore_index=True)
    panel = panel.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    panel["成交量"] = pd.to_numeric(panel["vol"], errors="coerce") * 100.0
    panel["成交额"] = pd.to_numeric(panel["amount"], errors="coerce") * 1000.0
    panel["换手率"] = pd.to_numeric(panel["turnover_rate"], errors="coerce")
    panel = panel.rename(
        columns={"open": "开盘", "close": "收盘", "high": "最高", "low": "最低"}
    )
    previous_close = panel.groupby("股票代码", sort=False)["收盘"].shift(1)
    panel["振幅"] = ((panel["最高"] - panel["最低"]) / previous_close * 100).round(2)
    panel["涨跌额"] = (panel["收盘"] - previous_close).round(6)
    panel["涨跌幅"] = ((panel["收盘"] / previous_close - 1) * 100).round(6)

    price_lookup = panel.set_index(["股票代码", "日期"])
    labels = np.full(len(panel), np.nan, dtype=float)
    entry_failed = np.zeros(len(panel), dtype=bool)
    for position, row in enumerate(panel.itertuples(index=False)):
        day_pos = calendar_position.get(row.日期)
        if day_pos is None or day_pos + 5 >= len(calendar):
            continue
        entry_day = calendar[day_pos + 1]
        exit_day = calendar[day_pos + 5]
        try:
            entry = price_lookup.loc[(row.股票代码, entry_day)]
            exit_row = price_lookup.loc[(row.股票代码, exit_day)]
        except KeyError:
            continue
        if bool(entry["is_suspended"]) or not np.isfinite(entry["开盘"]) or entry["开盘"] <= 0:
            labels[position] = 0.0
            entry_failed[position] = True
        elif np.isfinite(exit_row["收盘"]):
            labels[position] = (exit_row["收盘"] - entry["开盘"]) / entry["开盘"]

    panel["label"] = labels
    panel["label_entry_failed"] = entry_failed
    panel["source"] = "tushare_pit"
    signal_start = pd.Timestamp(signal_start).normalize()
    signal_end = pd.Timestamp(signal_end).normalize()
    panel["in_signal_range"] = panel["日期"].between(signal_start, signal_end)
    return panel[MODEL_COLUMNS + ["in_signal_range"]]


def point_in_time_cross_sectional_rank(
    frame: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Rank all histories against that day's true tradable member distribution."""
    result = frame.copy()
    member = (
        result["is_member"].fillna(False).astype(bool)
        if "is_member" in result
        else pd.Series(True, index=result.index)
    )
    tradable = (
        result["is_tradable"].fillna(False).astype(bool)
        if "is_tradable" in result
        else pd.Series(True, index=result.index)
    )
    candidate = member & tradable
    ranked = pd.DataFrame(index=result.index, columns=feature_columns, dtype=float)
    for _, day in result.groupby("日期", sort=False):
        day_candidate = candidate.loc[day.index]
        for column in feature_columns:
            values = pd.to_numeric(day[column], errors="coerce")
            reference = np.sort(values[day_candidate & values.notna()].to_numpy())
            if len(reference) == 0:
                continue
            valid = values.notna()
            ranked.loc[day.index[valid], column] = (
                np.searchsorted(reference, values[valid].to_numpy(), side="right")
                / len(reference)
            )
    result[feature_columns] = ranked[feature_columns]
    return result


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
