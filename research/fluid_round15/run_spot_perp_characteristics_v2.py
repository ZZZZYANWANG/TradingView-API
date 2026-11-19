from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

import run_spot_perp_characteristics as base

HF_REPO = "ismatsamadov/binance-usdt-archive"


def resolve_hf_file(files: set[str], config: str, symbol: str) -> str:
    exact = f"{config}/{symbol}.parquet"
    if exact in files:
        return exact
    matches = [name for name in files if config in name and name.endswith(f"/{symbol}.parquet")]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"cannot resolve {config}/{symbol}: {matches[:5]}")


def load_hf_external() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    api = HfApi()
    info = api.dataset_info(HF_REPO)
    revision = str(info.sha)
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset", revision=revision))
    metrics: dict[str, pd.DataFrame] = {}
    funding: dict[str, pd.DataFrame] = {}
    resolved: dict[str, dict[str, str]] = {}
    for symbol in base.SYMBOLS:
        metric_name = resolve_hf_file(files, "um_metrics", symbol)
        funding_name = resolve_hf_file(files, "um_funding", symbol)
        resolved[symbol] = {"metrics": metric_name, "funding": funding_name}
        metric_path = hf_hub_download(repo_id=HF_REPO, filename=metric_name, repo_type="dataset", revision=revision)
        funding_path = hf_hub_download(repo_id=HF_REPO, filename=funding_name, repo_type="dataset", revision=revision)

        m = pd.read_parquet(metric_path)
        m.columns = [str(column).strip().lower() for column in m.columns]
        aliases = {
            "sum_open_interest": "open_interest",
            "sum_open_interest_value": "open_interest_value",
            "top_trader_account_ratio": "top_count_ratio",
            "count_toptrader_long_short_ratio": "top_count_ratio",
            "top_trader_position_ratio": "top_position_ratio",
            "sum_toptrader_long_short_ratio": "top_position_ratio",
            "count_long_short_ratio": "account_ratio",
            "taker_buy_sell_ratio": "taker_ratio",
            "sum_taker_long_short_vol_ratio": "taker_ratio",
        }
        m = m.rename(columns=aliases)
        required = [
            "time", "open_interest", "open_interest_value", "top_count_ratio",
            "top_position_ratio", "account_ratio", "taker_ratio",
        ]
        for column in required[1:]:
            if column not in m:
                m[column] = np.nan
        m["time"] = pd.to_datetime(m["time"], utc=True, errors="coerce")
        for column in required[1:]:
            m[column] = pd.to_numeric(m[column], errors="coerce")
        m = (
            m.dropna(subset=["time", "open_interest", "open_interest_value"])
            .set_index("time")[required[1:]]
            .sort_index()
            .resample("1h")
            .last()
            .dropna(subset=["open_interest", "open_interest_value"])
            .reset_index()
        )
        m["time"] = m["time"] + pd.Timedelta(hours=1)
        metrics[symbol] = m

        f = pd.read_parquet(funding_path)
        f.columns = [str(column).strip().lower() for column in f.columns]
        time_col = next((column for column in ["time", "calc_time", "funding_time"] if column in f), None)
        rate_col = next((column for column in ["funding_rate", "last_funding_rate", "fundingrate"] if column in f), None)
        if time_col is None or rate_col is None:
            raise RuntimeError(f"funding columns missing for {symbol}: {f.columns.tolist()}")
        f["time"] = pd.to_datetime(f[time_col], utc=True, errors="coerce")
        f["funding_rate"] = pd.to_numeric(f[rate_col], errors="coerce")
        f = f[["time", "funding_rate"]].dropna().sort_values("time")
        f["time"] = f["time"] + pd.Timedelta(hours=1)
        funding[symbol] = f
    return metrics, funding, {"repo": HF_REPO, "revision": revision, "resolved": resolved}


def patched_load_public_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    tasks = [(symbol, period, venue) for symbol in base.SYMBOLS for period in base.MONTHS for venue in ["perp", "spot"]]
    buckets: dict[str, list[pd.DataFrame]] = {"perp": [], "spot": []}
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        jobs = {
            pool.submit(base.download_kline, symbol, period, venue): (symbol, period, venue)
            for symbol, period, venue in tasks
        }
        for count, future in enumerate(as_completed(jobs), 1):
            symbol, period, venue = jobs[future]
            frame, error = future.result()
            if frame is not None and not frame.empty:
                buckets[venue].append(frame)
            elif error != "404":
                errors.append({"symbol": symbol, "month": str(period), "venue": venue, "error": str(error)})
            if count % 250 == 0:
                print(f"kline_downloads={count}/{len(tasks)} perp={len(buckets['perp'])} spot={len(buckets['spot'])}", flush=True)
    if not buckets["perp"] or not buckets["spot"]:
        raise RuntimeError("spot or perpetual kline data missing")

    perp = pd.concat(buckets["perp"], ignore_index=True).drop_duplicates(["symbol", "time"], keep="last")
    spot = pd.concat(buckets["spot"], ignore_index=True).drop_duplicates(["symbol", "time"], keep="last")
    spot = spot.rename(columns={
        column: f"{column}_spot"
        for column in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"]
    })
    panel = perp.merge(spot, on=["time", "symbol"], how="inner", validate="one_to_one")
    metrics, funding, external_audit = load_hf_external()

    merged_parts: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=False):
        g = group.sort_values("time").copy()
        m = metrics[symbol].sort_values("time")
        g = pd.merge_asof(g, m, on="time", direction="backward", tolerance=pd.Timedelta(hours=2))
        f = funding[symbol].sort_values("time")
        g = pd.merge_asof(g, f, on="time", direction="backward", tolerance=pd.Timedelta(hours=12))
        merged_parts.append(g)
    panel = pd.concat(merged_parts, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)
    audit = {
        "requested_kline_downloads": len(tasks),
        "usable_parts": {key: len(value) for key, value in buckets.items()},
        "errors": errors[:100],
        "rows": int(len(panel)),
        "symbols": sorted(panel["symbol"].unique().tolist()),
        "min_time": str(panel["time"].min()),
        "max_time": str(panel["time"].max()),
        "metric_coverage": int(panel["open_interest"].notna().sum()),
        "funding_coverage": int(panel["funding_rate"].notna().sum()),
        "external": external_audit,
    }
    return panel, audit


base.load_public_data = patched_load_public_data

if __name__ == "__main__":
    base.main()
