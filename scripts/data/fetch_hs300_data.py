#!/usr/bin/env python3
"""从 akshare（新浪数据源）下载沪深300成分股近4年后复权日线数据。

输出格式与 data/stock_data.csv 一致：
股票代码,日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌额,换手率,涨跌幅

使用新浪接口（stock_zh_a_daily），避免东方财富限流。
支持 --resume 断点续传。
"""

import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

# ---------- 配置 ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STOCK_LIST = DATA_DIR / "hs300_stock_list.csv"
OUTPUT_FILE = DATA_DIR / "hs300_4y_data.csv"
PAUSE_MIN = 1.5
PAUSE_MAX = 3.0
MAX_RETRY = 4
RETRY_WAITS = [5, 15, 30, 60]
COOLDOWN_THRESHOLD = 5
COOLDOWN_SECONDS = 60

# 日期范围：昨天往前4年
YESTERDAY = date.today() - timedelta(days=1)
START_DATE = YESTERDAY.replace(year=YESTERDAY.year - 4)
START_STR = START_DATE.strftime("%Y%m%d")
END_STR = YESTERDAY.strftime("%Y%m%d")

OUTPUT_COLS = ["股票代码", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]


def load_stock_codes(path: Path) -> list[tuple[str, str]]:
    """读取成分股列表，返回 (sina_symbol, pure_code) 元组列表。
    新浪格式: sh600000 / sz000001
    """
    df = pd.read_csv(path, dtype=str)
    codes = []
    seen = set()
    for raw in df["code"].dropna():
        raw = raw.strip().lower()
        # hs300_stock_list.csv 里是 sh.600000 格式
        if "." in raw:
            prefix, digits = raw.split(".", 1)
        else:
            digits = raw
            prefix = "sh" if digits.startswith(("5", "6", "9")) else "sz"
        digits = "".join(c for c in digits if c.isdigit()).zfill(6)
        if digits not in seen:
            seen.add(digits)
            sina_symbol = f"{prefix}{digits}"
            codes.append((sina_symbol, digits))
    return codes


def fetch_stock(sina_symbol: str, pure_code: str) -> pd.DataFrame:
    """获取单只股票的后复权日线数据（新浪源）。"""
    df = ak.stock_zh_a_daily(
        symbol=sina_symbol,
        start_date=START_STR,
        end_date=END_STR,
        adjust="hfq",
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    # 新浪返回列: date, open, high, low, close, volume, amount, outstanding_share, turnover
    df = df.sort_values("date").reset_index(drop=True)

    # 计算 preclose（前一日收盘价）
    df["preclose"] = df["close"].shift(1)
    # 第一行没有 preclose，用 open 近似（振幅/涨跌额会略有偏差）
    df["preclose"] = df["preclose"].fillna(df["open"])

    # 振幅 = (最高 - 最低) / 前收盘 * 100
    df["振幅"] = ((df["high"] - df["low"]) / df["preclose"] * 100).round(2)
    # 涨跌额 = 收盘 - 前收盘
    df["涨跌额"] = (df["close"] - df["preclose"]).round(2)
    # 涨跌幅 = (收盘 - 前收盘) / 前收盘 * 100
    df["涨跌幅"] = ((df["close"] - df["preclose"]) / df["preclose"] * 100).round(4)
    # 换手率: sina 给的是小数比例，转为百分比
    df["换手率"] = (df["turnover"] * 100).round(4)

    # 组装输出（必须指定 index，否则标量列与 Series 索引不对齐）
    out = pd.DataFrame(index=df.index)
    out["股票代码"] = pure_code
    out["日期"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d 00:00:00")
    out["开盘"] = df["open"]
    out["收盘"] = df["close"]
    out["最高"] = df["high"]
    out["最低"] = df["low"]
    out["成交量"] = df["volume"]
    out["成交额"] = df["amount"]
    out["振幅"] = df["振幅"]
    out["涨跌额"] = df["涨跌额"]
    out["换手率"] = df["换手率"]
    out["涨跌幅"] = df["涨跌幅"]

    return out[OUTPUT_COLS]


def load_completed(output_path: Path) -> set[str]:
    """从已有输出中读取已完成的股票代码。"""
    if not output_path.exists():
        return set()
    try:
        df = pd.read_csv(output_path, dtype={"股票代码": str}, usecols=["股票代码"])
        return set(df["股票代码"].str.zfill(6).unique())
    except Exception:
        return set()


def main():
    resume = "--resume" in sys.argv

    all_codes = load_stock_codes(STOCK_LIST)
    completed = load_completed(OUTPUT_FILE) if resume else set()

    if completed:
        all_codes = [(s, c) for s, c in all_codes if c not in completed]
        print(f"[断点续传] 已完成 {len(completed)} 只，剩余 {len(all_codes)} 只")

    print(f"待下载: {len(all_codes)} 只股票")
    print(f"日期范围: {START_DATE} ~ {YESTERDAY}")
    print(f"复权方式: 后复权（新浪源）")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"请求间隔: {PAUSE_MIN}~{PAUSE_MAX}s")
    print("-" * 50, flush=True)

    all_frames = []
    if resume and OUTPUT_FILE.exists():
        all_frames.append(pd.read_csv(OUTPUT_FILE, dtype={"股票代码": str}))

    failed = []
    consecutive_failures = 0

    for i, (sina_sym, pure_code) in enumerate(all_codes, 1):
        success = False
        for attempt in range(MAX_RETRY):
            try:
                df = fetch_stock(sina_sym, pure_code)
                all_frames.append(df)
                print(f"[{i}/{len(all_codes)}] {pure_code}: {len(df)} 行", flush=True)
                success = True
                consecutive_failures = 0
                break
            except Exception as e:
                wait = RETRY_WAITS[min(attempt, len(RETRY_WAITS) - 1)]
                print(f"[{i}/{len(all_codes)}] {pure_code}: 第{attempt+1}次失败({e})，等待{wait}s...", flush=True)
                time.sleep(wait)

        if not success:
            failed.append(pure_code)
            consecutive_failures += 1
            print(f"[{i}/{len(all_codes)}] {pure_code}: 最终失败", file=sys.stderr, flush=True)

            if consecutive_failures >= COOLDOWN_THRESHOLD:
                print(f"[冷却] 连续失败{consecutive_failures}次，暂停{COOLDOWN_SECONDS}s...", flush=True)
                time.sleep(COOLDOWN_SECONDS)
                consecutive_failures = 0

        # 每20只保存一次中间结果
        if i % 20 == 0 and all_frames:
            tmp = pd.concat(all_frames, ignore_index=True)
            tmp["股票代码"] = tmp["股票代码"].astype(str).str.zfill(6)
            tmp = tmp.sort_values(["股票代码", "日期"]).drop_duplicates(["股票代码", "日期"], keep="last")
            tmp.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
            print(f"  [中间保存] {tmp['股票代码'].nunique()} 只股票, {len(tmp)} 行", flush=True)

        time.sleep(random.uniform(PAUSE_MIN, PAUSE_MAX))

    # 最终保存
    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)
        result["股票代码"] = result["股票代码"].astype(str).str.zfill(6)
        result = result.sort_values(["股票代码", "日期"]).drop_duplicates(["股票代码", "日期"], keep="last").reset_index(drop=True)
        result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

        print("-" * 50)
        print(f"完成！共 {len(result)} 行，{result['股票代码'].nunique()} 只股票")
        print(f"保存至: {OUTPUT_FILE}")
    else:
        print("没有获取到任何数据。", file=sys.stderr)

    if failed:
        print(f"失败 {len(failed)} 只: {failed}")
        print("可用 --resume 参数断点续传补全。")


if __name__ == "__main__":
    main()
