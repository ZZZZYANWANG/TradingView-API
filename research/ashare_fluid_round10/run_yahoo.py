from __future__ import annotations

import importlib.util
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

SCRIPT = Path("research/ashare_fluid_round10/run.py")
OUT = Path("research_outputs/ashare_fluid_round10_yahoo")
RECEIPT_URL = "https://raw.githubusercontent.com/ZZZZYANWANG/TradingView-API/research-fluid-stock-graph-v3-20261118/research_outputs/fluid_stock_graph_v3/persistent_session_download_receipt.json"
START = pd.Timestamp("2018-01-01", tz="UTC")
END = pd.Timestamp("2026-11-16", tz="UTC")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36"}


def load_module():
    spec = importlib.util.spec_from_file_location("ashare_cascade_core", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import cascade core")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_code_receipt() -> list[str]:
    response = requests.get(RECEIPT_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    codes = sorted({str(item["code"]) for item in payload if item.get("ok")})
    if len(codes) < 100:
        raise RuntimeError(f"historical-membership receipt too small: {len(codes)}")
    return codes


def yahoo_symbol(code: str) -> str:
    market, number = code.split(".")
    return f"{number}.SS" if market.lower() == "sh" else f"{number}.SZ"


def download_one(code: str) -> tuple[str, pd.DataFrame | None, str | None]:
    symbol = yahoo_symbol(code)
    params = {
        "period1": int(START.timestamp()),
        "period2": int(END.timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    error = ""
    for attempt in range(7):
        host = "query1.finance.yahoo.com" if attempt % 2 == 0 else "query2.finance.yahoo.com"
        url = f"https://{host}/v8/finance/chart/{symbol}"
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=35)
            if response.status_code in {429, 502, 503, 504}:
                error = f"http_{response.status_code}"
                time.sleep(1.5 * (attempt + 1))
                continue
            response.raise_for_status()
            result = response.json().get("chart", {}).get("result")
            if not result:
                return code, None, "empty_result"
            root = result[0]
            timestamps = root.get("timestamp") or []
            quote = (root.get("indicators", {}).get("quote") or [{}])[0]
            adj = (root.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
            if not timestamps or adj is None:
                return code, None, "missing_timestamp_or_adjclose"
            frame = pd.DataFrame(
                {
                    "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
                    "open_raw": quote.get("open"),
                    "high_raw": quote.get("high"),
                    "low_raw": quote.get("low"),
                    "close_raw": quote.get("close"),
                    "volume": quote.get("volume"),
                    "adjclose": adj,
                }
            )
            for column in ["open_raw", "high_raw", "low_raw", "close_raw", "volume", "adjclose"]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            adjustment = frame["adjclose"] / frame["close_raw"]
            frame["open"] = frame["open_raw"] * adjustment
            frame["high"] = frame["high_raw"] * adjustment
            frame["low"] = frame["low_raw"] * adjustment
            frame["close"] = frame["adjclose"]
            typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
            frame["amount"] = typical * frame["volume"]
            rolling_volume = frame["volume"].shift(1).rolling(60, min_periods=20).median()
            frame["turnover"] = frame["volume"] / (rolling_volume + 1.0e-12)
            frame["code"] = code
            frame["market"] = code[:2].upper()
            frame["name"] = code
            frame = frame.replace([np.inf, -np.inf], np.nan)
            frame = frame.dropna(subset=["date", "open", "high", "low", "close", "volume", "amount"])
            frame = frame.loc[
                frame[["open", "high", "low", "close", "volume", "amount"]].gt(0).all(axis=1)
            ].copy()
            if len(frame) < 900:
                return code, None, f"short_history_{len(frame)}"
            return code, frame[["date", "open", "high", "low", "close", "volume", "amount", "turnover", "code", "market", "name"]], None
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}:{exc}"
            time.sleep(1.5 * (attempt + 1))
    return code, None, error


def load_yahoo() -> tuple[pd.DataFrame, dict[str, Any]]:
    codes = load_code_receipt()
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(download_one, code): code for code in codes}
        for index, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                _, frame, error = future.result()
            except Exception as exc:  # noqa: BLE001
                frame, error = None, f"{type(exc).__name__}:{exc}"
            if frame is not None and not frame.empty:
                frames.append(frame)
            else:
                errors.append({"code": code, "error": str(error)})
            if index % 20 == 0:
                print(f"yahoo downloads={index}/{len(codes)} usable={len(frames)}", flush=True)
    if len(frames) < 80:
        raise RuntimeError(f"insufficient Yahoo A-share histories: {len(frames)}; errors={errors[:10]}")
    panel = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    by_market = panel.groupby("market")["code"].nunique().to_dict()
    audit = {
        "source": "Yahoo Finance adjusted daily chart API",
        "universe_source": "historical CSI300 membership-frequency code receipt from successful Baostock intraday download",
        "universe_receipt_url": RECEIPT_URL,
        "point_in_time_membership": False,
        "requested": len(codes),
        "usable": panel["code"].nunique(),
        "usable_by_market": by_market,
        "rows": len(panel),
        "min_date": str(panel["date"].min().date()),
        "max_date": str(panel["date"].max().date()),
        "errors": errors,
        "turnover_proxy": "volume divided by strict-prior 60-session median volume",
        "limitations": [
            "historical-membership-frequency universe is fixed over the sample",
            "daily external replication does not replace the local one-minute path calculation",
            "Yahoo adjusted OHLC uses adjusted-close ratio applied to raw OHLC",
        ],
    }
    return panel, audit


def main() -> None:
    module = load_module()
    module.OUT = OUT
    module.OUT.mkdir(parents=True, exist_ok=True)
    module.load = load_yahoo
    module.main()


if __name__ == "__main__":
    main()
