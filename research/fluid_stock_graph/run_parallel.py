from __future__ import annotations

import importlib.util
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import baostock as bs
import pandas as pd

SCRIPT = Path("research/fluid_stock_graph/run_research.py")
CACHE = Path("research_outputs/fluid_stock_graph_parallel/cache")
OUTPUT = Path("research_outputs/fluid_stock_graph_parallel")
MAX_CODES = 100
WORKERS = 6


def load_module():
    spec = importlib.util.spec_from_file_location("fluid_graph_module", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load research module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MAX_CODES = MAX_CODES
    return module


def download_one(code: str) -> dict[str, object]:
    module = load_module()
    login = bs.login()
    if login.error_code != "0":
        return {"code": code, "ok": False, "error": login.error_msg}
    try:
        daily = module.download_daily(code)
        hourly = module.download_hourly(code)
    finally:
        bs.logout()
    if daily.empty or hourly.empty:
        return {"code": code, "ok": False, "error": "empty_daily_or_hourly"}
    CACHE.mkdir(parents=True, exist_ok=True)
    daily.to_pickle(CACHE / f"{code.replace('.', '_')}.daily.pkl")
    hourly.to_pickle(CACHE / f"{code.replace('.', '_')}.hourly.pkl")
    return {"code": code, "ok": True, "daily_rows": len(daily), "hourly_rows": len(hourly)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    module = load_module()
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    try:
        snapshots, membership_meta = module.load_membership_snapshots()
    finally:
        bs.logout()
    counts = Counter(code for members in snapshots.values() for code in members)
    codes = [code for code, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_CODES]]
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, code): code for code in codes}
        for index, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"code": code, "ok": False, "error": f"{type(exc).__name__}:{exc}"}
            results.append(result)
            if index % 10 == 0:
                good = sum(bool(row.get("ok")) for row in results)
                print(f"parallel downloads={index}/{len(codes)} usable={good}", flush=True)
    with (OUTPUT / "parallel_download_receipt.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    good_codes = {str(row["code"]) for row in results if row.get("ok")}
    if len(good_codes) < 60:
        raise RuntimeError(f"only {len(good_codes)} codes downloaded successfully")

    def cached_daily(code: str) -> pd.DataFrame:
        path = CACHE / f"{code.replace('.', '_')}.daily.pkl"
        return pd.read_pickle(path) if path.exists() else pd.DataFrame()

    def cached_hourly(code: str) -> pd.DataFrame:
        path = CACHE / f"{code.replace('.', '_')}.hourly.pkl"
        return pd.read_pickle(path) if path.exists() else pd.DataFrame()

    module.OUT = OUTPUT
    module.MAX_CODES = MAX_CODES
    module.load_membership_snapshots = lambda: (snapshots, membership_meta)
    module.download_daily = cached_daily
    module.download_hourly = cached_hourly
    module.main()


if __name__ == "__main__":
    main()
