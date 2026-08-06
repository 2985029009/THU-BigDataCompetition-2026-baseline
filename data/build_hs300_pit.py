#!/usr/bin/env python3
"""Build an auditable point-in-time CSI 300 research dataset.

The Tushare token is read exclusively from ``TUSHARE_TOKEN``.  It is never
accepted as a command-line argument and is never persisted.
"""

from __future__ import annotations

import argparse
import calendar as month_calendar
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code" / "src"
sys.path.insert(0, str(SRC))

from pit_data import (  # noqa: E402
    build_membership,
    build_model_data,
    normalize_ts_code,
    sha256_file,
    write_json,
)


INDEX_CODE = "000300.SH"
CSI_SEARCH_URL = (
    "https://www.csindex.com.cn/csindex-home/search/search-content"
    "?lang=cn&searchInput=%E6%B2%AA%E6%B7%B1300"
    "&pageNum={page}&pageSize={size}&sortField=date&dateRange=all&contentType=announcement"
)
CSI_DETAIL_URL = (
    "https://www.csindex.com.cn/csindex-home/announcement/queryAnnouncementById?id={id}"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124 Safari/537.36"
    )
}
LINKED_DELISTING_EVENTS = {
    # CSI notices define the index change as effective on the constituent's
    # delisting date.  The linked SSE notices provide that exact date.
    "600837": {
        "date": "2025-03-04",
        "remove": "600837.SH",
        "add": "601058.SH",
        "source": "https://www.sse.com.cn/disclosure/announcement/listing/stock/c/c_20250226_10773005.shtml",
    },
    "601989": {
        "date": "2025-09-05",
        "remove": "601989.SH",
        "add": "601298.SH",
        "source": "https://www.sse.com.cn/disclosure/announcement/listing/stock/c/c_20250829_10790128.shtml",
    },
}
SNAPSHOT_VALIDATED_OFFICIAL_EVENTS = [
    {
        "effective_date": "2025-12-15",
        "source": (
            "https://www.csindex.com.cn/csindex-home/announcement/"
            "queryAnnouncementById?id=3006000"
        ),
        "expected_changes": 11,
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建沪深300时点真实(PIT)研究数据")
    parser.add_argument("--start-date", default="2022-07-25")
    parser.add_argument("--end-date", default="2026-07-24")
    parser.add_argument("--output-dir", type=Path, default=Path("data/hs300_pit"))
    parser.add_argument("--resume", action="store_true", help="复用已保存的原始响应")
    parser.add_argument("--request-pause", type=float, default=0.12)
    args = parser.parse_args()
    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        parser.error("--start-date 不能晚于 --end-date")
    return args


def _safe_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    data_root = (ROOT / "data").resolve()
    if resolved == data_root or data_root not in resolved.parents:
        raise ValueError("PIT输出目录必须是项目 data 目录下的独立子目录")
    return resolved


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp):
    cursor = start.to_period("M")
    final = end.to_period("M")
    while cursor <= final:
        first = cursor.start_time.normalize()
        last_day = month_calendar.monthrange(first.year, first.month)[1]
        last = pd.Timestamp(first.year, first.month, last_day)
        yield first, last
        cursor += 1


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _cached_query(path: Path, resume: bool, query_fn) -> pd.DataFrame:
    if resume and path.exists():
        return pd.read_csv(path, dtype=str)
    frame = query_fn()
    if frame is None:
        frame = pd.DataFrame()
    _atomic_csv(frame, path)
    return frame


def _preflight(pro) -> dict:
    checks = {
        "trade_cal": lambda: pro.trade_cal(
            exchange="SSE", start_date="20240701", end_date="20240705"
        ),
        "index_weight": lambda: pro.index_weight(
            index_code=INDEX_CODE, start_date="20240701", end_date="20240731"
        ),
        "daily": lambda: pro.daily(ts_code="000001.SZ", trade_date="20240701"),
        "adj_factor": lambda: pro.adj_factor(ts_code="000001.SZ", trade_date="20240701"),
        "daily_basic": lambda: pro.daily_basic(
            ts_code="000001.SZ", trade_date="20240701", fields="ts_code,trade_date,turnover_rate"
        ),
        "suspend_d": lambda: pro.suspend_d(
            ts_code="000001.SZ", start_date="20240701", end_date="20240705"
        ),
    }
    status = {}
    for name, call in checks.items():
        try:
            value = call()
            status[name] = {"ok": True, "rows": int(len(value))}
        except Exception as exc:
            status[name] = {"ok": False, "error": str(exc)}
    failed = [name for name, item in status.items() if not item["ok"]]
    if failed:
        raise PermissionError(f"Tushare必要接口预检失败: {', '.join(failed)}")
    return status


def _download_calendar(pro, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    broad_start = (start - pd.DateOffset(years=1)).strftime("%Y%m%d")
    broad_end = (end + pd.DateOffset(months=1)).strftime("%Y%m%d")
    frame = pro.trade_cal(exchange="SSE", start_date=broad_start, end_date=broad_end)
    opens = pd.to_datetime(frame.loc[frame["is_open"].astype(int).eq(1), "cal_date"])
    opens = pd.DatetimeIndex(opens).normalize().unique().sort_values()
    before = opens[opens < start]
    if len(before) < 118:
        raise ValueError(f"预热交易日不足118天，实际只有{len(before)}天")
    return opens


def _download_snapshots(pro, raw_dir: Path, start: pd.Timestamp, end: pd.Timestamp, resume: bool):
    frames = []
    query_start = (start - pd.DateOffset(months=2)).normalize()
    for first, last in _month_ranges(query_start, end):
        path = raw_dir / "index_weight" / f"{first:%Y%m}.csv"
        frame = _cached_query(
            path,
            resume,
            lambda first=first, last=last: pro.index_weight(
                index_code=INDEX_CODE,
                start_date=first.strftime("%Y%m%d"),
                end_date=last.strftime("%Y%m%d"),
            ),
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("index_weight 未返回任何沪深300历史成分数据")
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(["trade_date", "con_code"], keep="last")
    return result


def _request_json(session: requests.Session, url: str) -> dict:
    response = session.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return _repair_csi_text(response.json())


def _repair_csi_text(value):
    """Repair UTF-8 text that the CSI endpoint has decoded as GBK."""
    if isinstance(value, dict):
        return {key: _repair_csi_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_repair_csi_text(item) for item in value]
    if not isinstance(value, str):
        return value
    try:
        repaired = value.encode("gbk").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    markers = ("关于", "调整", "公告", "生效", "实施", "调入", "调出")
    return repaired if any(marker in repaired for marker in markers) else value


def _effective_date(content: str, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    plain = re.sub(r"<[^>]+>", " ", content)
    # Recent CSI notices sometimes insert formatting spaces inside numbers
    # (for example ``202 5 年``).  Remove only digit-to-digit whitespace.
    plain = re.sub(r"(?<=\d)\s+(?=\d)", "", plain)
    patterns = [
        r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日[^。]{0,30}(?:生效|实施)",
        r"(?:生效|实施)[^。]{0,30}(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
    ]
    match = next((re.search(pattern, plain) for pattern in patterns if re.search(pattern, plain)), None)
    if match is None:
        raise ValueError("公告正文中没有可审计的明确生效日期")
    stated = pd.Timestamp(*map(int, match.groups())).normalize()
    after_close = bool(re.search(r"(?:收市|盘后|市后)[^。]{0,12}(?:生效|实施)", plain))
    if after_close:
        later = trading_days[trading_days > stated]
        if len(later) == 0:
            raise ValueError(f"无法找到 {stated.date()} 收市后的下一交易日")
        return later[0]
    if stated not in trading_days:
        later = trading_days[trading_days > stated]
        if len(later) == 0:
            raise ValueError(f"生效日 {stated.date()} 后没有交易日")
        return later[0]
    return stated


def _linked_delisting_date(
    content: str, trading_days: pd.DatetimeIndex
) -> tuple[pd.Timestamp, str, dict] | None:
    plain = re.sub(r"<[^>]+>", " ", content)
    if "退市日起" not in plain:
        return None
    for code, event in LINKED_DELISTING_EVENTS.items():
        if code not in plain:
            continue
        effective = pd.Timestamp(event["date"]).normalize()
        if effective not in trading_days:
            raise ValueError(f"交易所退市生效日 {effective.date()} 不是交易日")
        return effective, event["source"], event
    return None


def _attachment_events(
    content: bytes,
    suffix: str,
    effective_date: pd.Timestamp,
    source_url: str,
) -> pd.DataFrame:
    if suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        section = re.search(
            r"沪深\s*300\s*指数样本调整名单[：:]?(.*?)(?=中证\s*500\s*指数样本调整名单|$)",
            text,
            flags=re.S,
        )
        if section is None:
            return pd.DataFrame()
        rows = []
        for line in section.group(1).splitlines():
            codes = re.findall(r"(?<!\d)(\d{6})(?!\d)", line)
            if len(codes) < 2:
                continue
            for code, action in ((codes[0], "remove"), (codes[1], "add")):
                rows.append(
                    {
                        "effective_date": effective_date,
                        "con_code": normalize_ts_code(code),
                        "action": action,
                        "source_announcement": source_url,
                    }
                )
        return pd.DataFrame(rows)
    if suffix.lower() not in {".xls", ".xlsx"}:
        return pd.DataFrame()
    sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
    rows = []
    for sheet_name, action in (("调入", "add"), ("调出", "remove")):
        candidates = [frame for name, frame in sheets.items() if sheet_name in str(name)]
        for frame in candidates:
            code_col = next((col for col in frame.columns if "证券代码" in str(col)), None)
            index_col = next((col for col in frame.columns if "指数代码" in str(col)), None)
            if code_col is None:
                continue
            selected = frame
            if index_col is not None:
                normalized_index = (
                    selected[index_col]
                    .astype(str)
                    .str.extract(r"(\d+)", expand=False)
                    .str.zfill(6)
                )
                selected = selected[
                    normalized_index.isin({"000300", "399300"})
                ]
            for code in selected[code_col].dropna():
                rows.append(
                    {
                        "effective_date": effective_date,
                        "con_code": normalize_ts_code(code),
                        "action": action,
                        "source_announcement": source_url,
                    }
                )
    return pd.DataFrame(rows)


def _html_table_events(
    html: str, effective_date: pd.Timestamp, source_url: str
) -> pd.DataFrame:
    rows = []
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return pd.DataFrame()
    for table in tables:
        if table.shape[1] < 4:
            continue
        text = " ".join(map(str, table.astype(str).values.flatten()))
        if "调入" not in text or "调出" not in text:
            continue
        for column_index, action in ((0, "remove"), (2, "add")):
            for value in table.iloc[:, column_index].astype(str):
                match = re.search(r"(?<!\d)(\d{6})(?!\d)", value)
                if match:
                    rows.append(
                        {
                            "effective_date": effective_date,
                            "con_code": normalize_ts_code(match.group(1)),
                            "action": action,
                            "source_announcement": source_url,
                        }
                    )
        if rows:
            break
    return pd.DataFrame(rows)


def _download_events(
    raw_dir: Path,
    trading_days: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    resume: bool,
) -> pd.DataFrame:
    session = requests.Session()
    search_path = raw_dir / "announcements" / "search.json"
    search_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and search_path.exists():
        search = _repair_csi_text(
            json.loads(search_path.read_text(encoding="utf-8"))
        )
    else:
        first = _request_json(session, CSI_SEARCH_URL.format(page=1, size=5))
        total = int(first.get("total", len(first.get("data", []))))
        search = _request_json(session, CSI_SEARCH_URL.format(page=1, size=max(total, 5)))
        write_json(search_path, search)

    event_frames = []
    parse_errors = []
    for item in search.get("data", []):
        # The CSI search API historically used ``title`` but currently returns
        # ``headline``.  The search endpoint is already scoped to CSI 300;
        # retain a lightweight numeric guard because highlighted titles contain
        # HTML and older responses have occasionally been mojibake-encoded.
        title = str(item.get("headline") or item.get("title") or "")
        if item.get("itemType") not in (None, "announcement") or "300" not in title:
            continue
        item_date = pd.to_datetime(item.get("itemDate"), errors="coerce")
        if pd.notna(item_date) and (
            item_date < start - pd.DateOffset(months=6)
            or item_date > end + pd.DateOffset(months=1)
        ):
            continue
        announcement_id = str(item["id"])
        detail_path = raw_dir / "announcements" / f"{announcement_id}.json"
        if resume and detail_path.exists():
            payload = _repair_csi_text(
                json.loads(detail_path.read_text(encoding="utf-8"))
            )
        else:
            payload = _request_json(
                session, CSI_DETAIL_URL.format(id=announcement_id)
            )
            write_json(detail_path, payload)
        detail = payload.get("data", payload)
        content = str(detail.get("content", ""))
        source_url = CSI_DETAIL_URL.format(id=announcement_id)
        try:
            effective = _effective_date(content, trading_days)
        except ValueError as exc:
            linked = _linked_delisting_date(content, trading_days)
            if linked is None:
                parse_errors.append(
                    {"id": announcement_id, "title": title, "error": str(exc)}
                )
                continue
            effective, linked_source, linked_event = linked
            source_url = f"{source_url};{linked_source}"
        else:
            linked_event = None
        if effective < start - pd.DateOffset(months=3) or effective > end:
            continue

        parsed = pd.DataFrame()
        for enclosure in detail.get("enclosureList", []) or []:
            file_url = enclosure.get("fileUrl")
            if not file_url:
                continue
            suffix = Path(file_url.split("?", 1)[0]).suffix or ".bin"
            attachment_path = raw_dir / "announcements" / f"{announcement_id}{suffix}"
            if resume and attachment_path.exists():
                body = attachment_path.read_bytes()
            else:
                try:
                    response = session.get(file_url, headers=HEADERS, timeout=30)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    parse_errors.append(
                        {
                            "id": announcement_id,
                            "title": title,
                            "error": f"附件下载失败: {exc}",
                        }
                    )
                    continue
                body = response.content
                attachment_path.write_bytes(body)
            candidate = _attachment_events(body, suffix, effective, source_url)
            if not candidate.empty:
                parsed = pd.concat([parsed, candidate], ignore_index=True)
        if parsed.empty:
            parsed = _html_table_events(content, effective, source_url)
        if parsed.empty and linked_event is not None:
            parsed = pd.DataFrame(
                [
                    {
                        "effective_date": effective,
                        "con_code": linked_event["remove"],
                        "action": "remove",
                        "source_announcement": source_url
                        + ";tushare:index_weight:next_snapshot",
                    },
                    {
                        "effective_date": effective,
                        "con_code": linked_event["add"],
                        "action": "add",
                        "source_announcement": source_url
                        + ";tushare:index_weight:next_snapshot",
                    },
                ]
            )
        if parsed.empty:
            parse_errors.append(
                {"id": announcement_id, "title": title, "error": "未解析到调入/调出代码"}
            )
        else:
            event_frames.append(parsed)

    write_json(raw_dir / "announcements" / "parse_report.json", {"errors": parse_errors})
    if not event_frames:
        raise ValueError("没有从中证官方公告解析出任何沪深300调整事件")
    events = pd.concat(event_frames, ignore_index=True)
    return events.drop_duplicates(["effective_date", "con_code", "action"])


def _complete_official_events_from_snapshots(
    events: pd.DataFrame, snapshots: pd.DataFrame
) -> pd.DataFrame:
    result = events.copy()
    weights = snapshots.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    weights["con_code"] = weights["con_code"].map(normalize_ts_code)
    for spec in SNAPSHOT_VALIDATED_OFFICIAL_EVENTS:
        effective = pd.Timestamp(spec["effective_date"])
        if (
            not result.empty
            and pd.to_datetime(result["effective_date"]).eq(effective).any()
        ):
            continue
        before_dates = weights.loc[weights["trade_date"] < effective, "trade_date"]
        after_dates = weights.loc[weights["trade_date"] >= effective, "trade_date"]
        if before_dates.empty or after_dates.empty:
            raise ValueError(f"{effective.date()} 前后缺少Tushare月度快照")
        before_date = before_dates.max()
        after_date = after_dates.min()
        before = set(
            weights.loc[weights["trade_date"].eq(before_date), "con_code"]
        )
        after = set(weights.loc[weights["trade_date"].eq(after_date), "con_code"])
        removed, added = sorted(before - after), sorted(after - before)
        expected = int(spec["expected_changes"])
        if len(removed) != expected or len(added) != expected:
            raise ValueError(
                f"{effective.date()} 官方公告称更换{expected}只，"
                f"快照差异为调出{len(removed)}只/调入{len(added)}只"
            )
        source = (
            f"{spec['source']};tushare:index_weight:"
            f"{before_date:%Y-%m-%d}->{after_date:%Y-%m-%d}"
        )
        supplement = pd.DataFrame(
            [
                *[
                    {
                        "effective_date": effective,
                        "con_code": code,
                        "action": "remove",
                        "source_announcement": source,
                    }
                    for code in removed
                ],
                *[
                    {
                        "effective_date": effective,
                        "con_code": code,
                        "action": "add",
                        "source_announcement": source,
                    }
                    for code in added
                ],
            ]
        )
        result = pd.concat([result, supplement], ignore_index=True)
    return result.drop_duplicates(["effective_date", "con_code", "action"])


def _download_market(
    pro,
    codes: list[str],
    raw_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    resume: bool,
    pause: float,
) -> pd.DataFrame:
    frames = []
    for number, code in enumerate(codes, start=1):
        path = raw_dir / "market" / f"{code.replace('.', '_')}.csv"
        if resume and path.exists():
            merged = pd.read_csv(path, dtype={"ts_code": str, "trade_date": str})
        else:
            daily = pro.daily(
                ts_code=code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
            factor = pro.adj_factor(
                ts_code=code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
            basic = pro.daily_basic(
                ts_code=code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                fields="ts_code,trade_date,turnover_rate",
            )
            suspend = pro.suspend_d(
                ts_code=code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
            merged = daily.merge(factor, on=["ts_code", "trade_date"], how="left")
            merged = merged.merge(basic, on=["ts_code", "trade_date"], how="left")
            if not suspend.empty and "trade_date" in suspend:
                suspend_dates = set(suspend["trade_date"].astype(str))
                merged["reported_suspend"] = merged["trade_date"].astype(str).isin(suspend_dates)
            else:
                merged["reported_suspend"] = False
            _atomic_csv(merged, path)
            time.sleep(pause)
        if merged.empty:
            raise ValueError(f"{code} 在请求区间内没有行情数据")
        frames.append(merged)
        print(f"[{number}/{len(codes)}] {code}: {len(merged)} 行", flush=True)
    return pd.concat(frames, ignore_index=True)


def _quality_report(
    model: pd.DataFrame,
    membership_audit: dict,
    current_codes: set[str],
) -> dict:
    candidate = model["is_member"].astype(bool)
    prices = model[["开盘", "收盘", "最高", "最低"]]
    union = set(model.loc[candidate, "股票代码"])
    historical_removed = sorted(union - current_codes)
    signal_members = model[candidate & model["in_signal_range"].astype(bool)]
    signal_counts = signal_members.groupby("日期")["股票代码"].nunique()
    allowed_unlabelled_dates = set(sorted(signal_members["日期"].unique())[-5:])
    unexpected_null_label = signal_members[
        signal_members["label"].isna()
        & ~signal_members["日期"].isin(allowed_unlabelled_dates)
    ]
    core_columns = [
        "开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率", "adj_factor"
    ]
    ohlc_invalid = (
        model["最高"].lt(model[["开盘", "收盘", "最低"]].max(axis=1))
        | model["最低"].gt(model[["开盘", "收盘", "最高"]].min(axis=1))
    )
    report = {
        "status": "passed",
        "membership": membership_audit,
        "rows": int(len(model)),
        "stocks": int(model["股票代码"].nunique()),
        "signal_rows": int((candidate & model["in_signal_range"]).sum()),
        "historical_removed_stock_count": len(historical_removed),
        "historical_removed_sample": historical_removed[:20],
        "duplicate_code_date": int(model.duplicated(["股票代码", "日期"]).sum()),
        "bad_signal_member_count_dates": {
            str(pd.Timestamp(day).date()): int(count)
            for day, count in signal_counts[signal_counts.ne(300)].items()
        },
        "unexpected_core_null_rows": int(model[core_columns].isna().any(axis=1).sum()),
        "ohlc_invalid_rows": int(ohlc_invalid.sum()),
        "negative_volume_rows": int(model["成交量"].lt(0).sum()),
        "nonpositive_price_rows": int(prices.le(0).any(axis=1).sum()),
        "candidate_null_label_rows": int(
            (candidate & model["in_signal_range"] & model["label"].isna()).sum()
        ),
        "unexpected_candidate_null_label_rows": int(len(unexpected_null_label)),
    }
    failures = []
    if membership_audit["status"] != "passed":
        failures.append("membership_reconciliation")
    if report["historical_removed_stock_count"] == 0:
        failures.append("no_historical_removed_stocks")
    if report["bad_signal_member_count_dates"]:
        failures.append("bad_signal_member_count_dates")
    for key in [
        "duplicate_code_date",
        "unexpected_core_null_rows",
        "ohlc_invalid_rows",
        "negative_volume_rows",
        "nonpositive_price_rows",
        "unexpected_candidate_null_label_rows",
    ]:
        if report[key]:
            failures.append(key)
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
    return report


def main() -> int:
    args = parse_args()
    output = _safe_output_dir(args.output_dir)
    raw_dir = output / "raw"
    output.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        print("缺少环境变量 TUSHARE_TOKEN；未启动下载。", file=sys.stderr)
        return 2

    import tushare as ts

    pro = ts.pro_api(token)
    try:
        preflight = _preflight(pro)
    except PermissionError as exc:
        write_json(output / "preflight_report.json", {"status": "failed", "error": str(exc)})
        print(str(exc), file=sys.stderr)
        return 3

    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize()
    trading_days = _download_calendar(pro, start, end)
    warmup_start = trading_days[trading_days < start][-118]
    later = trading_days[trading_days > end]
    requested_market_end = later[4] if len(later) >= 5 else trading_days[-1]

    # Membership is also required throughout the 118-day feature/sequence warmup;
    # otherwise pre-start cross-sectional ranks would have no point-in-time reference set.
    snapshots = _download_snapshots(pro, raw_dir, warmup_start, end, args.resume)
    events = _download_events(raw_dir, trading_days, warmup_start, end, args.resume)
    events = _complete_official_events_from_snapshots(events, snapshots)
    _atomic_csv(events, raw_dir / "announcements" / "parsed_events.csv")
    membership_days = trading_days[
        (trading_days >= warmup_start) & (trading_days <= end)
    ]
    intervals, daily_membership, membership_audit = build_membership(
        snapshots, events, membership_days, warmup_start, end
    )
    write_json(output / "membership_reconciliation.json", membership_audit)
    if membership_audit["status"] != "passed":
        print("成分重建与Tushare月度快照不一致，已停止生成正式数据。", file=sys.stderr)
        return 4

    codes = sorted(daily_membership["ts_code"].unique())
    market = _download_market(
        pro, codes, raw_dir, warmup_start, requested_market_end, args.resume, args.request_pause
    )
    actual_market_end = pd.to_datetime(market["trade_date"]).max().normalize()
    available_calendar = trading_days[trading_days <= actual_market_end]
    model = build_model_data(market, daily_membership, available_calendar, start, end)

    latest_snapshot_date = pd.to_datetime(snapshots["trade_date"]).max()
    current_codes = set(
        snapshots.loc[
            pd.to_datetime(snapshots["trade_date"]).eq(latest_snapshot_date), "con_code"
        ].map(lambda value: normalize_ts_code(value).split(".")[0])
    )
    quality = _quality_report(model, membership_audit, current_codes)
    write_json(output / "quality_report.json", quality)
    if quality["status"] != "passed":
        print(f"质量检查失败: {quality.get('failures', [])}", file=sys.stderr)
        return 5

    _atomic_csv(intervals, output / "membership_intervals.csv")
    _atomic_csv(daily_membership, output / "daily_membership.csv")
    _atomic_csv(model, output / "model_data.csv")
    manifest = {
        "status": "validated",
        "dataset": "CSI300 point-in-time",
        "index_code": INDEX_CODE,
        "signal_start": start.strftime("%Y-%m-%d"),
        "signal_end": end.strftime("%Y-%m-%d"),
        "warmup_start": warmup_start.strftime("%Y-%m-%d"),
        "requested_market_end": requested_market_end.strftime("%Y-%m-%d"),
        "actual_market_end": actual_market_end.strftime("%Y-%m-%d"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tushare_version": ts.__version__,
        "preflight": preflight,
        "sources": {
            "membership_snapshots": "Tushare index_weight",
            "membership_effective_dates": "CSI official adjustment announcements",
            "market": "Tushare daily/adj_factor/daily_basic/suspend_d",
        },
        "request_scope": {
            "index_weight": {
                "index_code": INDEX_CODE,
                "start_date": warmup_start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "frequency": "monthly",
            },
            "market": {
                "start_date": warmup_start.strftime("%Y-%m-%d"),
                "requested_end_date": requested_market_end.strftime("%Y-%m-%d"),
                "actual_end_date": actual_market_end.strftime("%Y-%m-%d"),
                "endpoints": ["daily", "adj_factor", "daily_basic", "suspend_d"],
            },
        },
        "raw_files": {},
        "files": {},
    }
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        item = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if path.suffix.lower() == ".csv":
            try:
                item["rows"] = int(sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1)
            except UnicodeDecodeError:
                pass
        manifest["raw_files"][str(path.relative_to(output)).replace("\\", "/")] = item
    for name in [
        "membership_intervals.csv",
        "daily_membership.csv",
        "model_data.csv",
        "quality_report.json",
        "membership_reconciliation.json",
    ]:
        path = output / name
        manifest["files"][name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    write_json(output / "manifest.json", manifest)
    print(f"PIT数据构建完成: {output / 'model_data.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
