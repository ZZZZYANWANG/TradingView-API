from __future__ import annotations

import importlib.util
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SOURCE = Path("research/fluid_round9/run.py")
spec = importlib.util.spec_from_file_location("fluid_round9_core", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load round9 core")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

OUT = Path("research_outputs/fluid_round9_stock_yahoo")
OUT.mkdir(parents=True, exist_ok=True)

START = int(pd.Timestamp("2012-01-01", tz="UTC").timestamp())
END = int(pd.Timestamp("2026-11-01", tz="UTC").timestamp())


def download_yahoo(ticker: str):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": START,
        "period2": END,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0 Trade-fluid-round9"}
    error = None
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=35)
            response.raise_for_status()
            result = response.json()["chart"]["result"][0]
            timestamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]
            adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])
            frame = pd.DataFrame({
                "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
                "open_raw": quote["open"],
                "high_raw": quote["high"],
                "low_raw": quote["low"],
                "close_raw": quote["close"],
                "close": adj,
                "volume": quote["volume"],
            })
            for col in ["open_raw", "high_raw", "low_raw", "close_raw", "close", "volume"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            ratio = frame["close"] / frame["close_raw"]
            frame["open"] = frame["open_raw"] * ratio
            frame["high"] = frame["high_raw"] * ratio
            frame["low"] = frame["low_raw"] * ratio
            frame = frame[["date", "open", "high", "low", "close", "volume"]]
            frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
            frame = frame.loc[(frame[["open", "high", "low", "close"]] > 0).all(axis=1) & (frame["volume"] > 0)]
            if len(frame) < 2000:
                raise ValueError(f"short_history={len(frame)}")
            frame["symbol"] = ticker
            frame["sector"] = core.STOCK_SECTORS[ticker]
            return ticker, frame, None
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}:{exc}"
            time.sleep(1.0 + attempt)
    return ticker, None, error


def load_yahoo_panel():
    frames = []
    errors = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(download_yahoo, ticker) for ticker in core.STOCK_SECTORS]
        for future in as_completed(futures):
            ticker, frame, error = future.result()
            if frame is not None:
                frames.append(frame)
            else:
                errors.append({"ticker": ticker, "error": error})
    if len(frames) < 35:
        raise RuntimeError(f"only {len(frames)} usable Yahoo histories")
    panel = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    metadata = {
        "provider": "Yahoo chart API",
        "requested": sorted(core.STOCK_SECTORS),
        "retained": sorted(panel["symbol"].unique().tolist()),
        "rows": int(len(panel)),
        "min_date": str(panel["date"].min()),
        "max_date": str(panel["date"].max()),
        "errors": errors,
        "universe_warning": "fixed current large-cap universe; survivorship selection remains",
    }
    return panel, metadata


try:
    raw, metadata = load_yahoo_panel()
    parts = []
    for symbol, frame in raw.groupby("symbol", sort=False):
        computed = core.compute_stock_symbol(frame)
        computed["symbol"] = symbol
        computed["sector"] = core.STOCK_SECTORS[symbol]
        parts.append(computed)
    panel = core.add_stock_cross_section(pd.concat(parts, ignore_index=True))
    panel["split"] = core.split_label(panel["date"], "stocks")
    selection = core.select_families(panel, core.STOCK_FAMILIES, core.STOCK_HORIZONS, "stocks")
    validation, holdout, promoted = core.validate_stocks(panel, selection)
    selection.to_csv(OUT / "STOCK_SELECTION.csv", index=False)
    validation.to_csv(OUT / "STOCK_VALIDATION.csv", index=False)
    holdout.to_csv(OUT / "STOCK_HOLDOUT.csv", index=False)
    summary = {
        "completed_utc": str(pd.Timestamp.now(tz="UTC")),
        "data": metadata,
        "validation_passers": int(validation["validation_pass"].sum()) if len(validation) else 0,
        "promoted": promoted,
        "protocol": {
            "development": "2012-2020",
            "validation": "2021-2023",
            "holdout": "2024-2026-10",
            "cost_bp_one_way": 10.0,
            "holdout_policy": "computed only after validation pass",
        },
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "PROMOTED.json").write_text(json.dumps(core.jsonable(promoted), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Stock Sector-Flow Fluid Validation — Yahoo Replication",
        "",
        "## Selection", "", selection.to_markdown(index=False), "",
        "## Validation", "", validation.to_markdown(index=False), "",
        "## Conditional holdout", "", holdout.to_markdown(index=False) if len(holdout) else "No validation passer; holdout remained sealed.", "",
        f"Promoted factors: {len(promoted)}",
        "",
        "This is a provider replication of the frozen Stooq specification, not a new search pass.",
        "The fixed current large-cap universe retains survivorship-selection risk.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (OUT / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print(failure["traceback"])
    raise
