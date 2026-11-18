from __future__ import annotations

import importlib.util
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import baostock as bs
import pandas as pd

V2_SCRIPT = Path("research/fluid_stock_graph/run_intraday_v2.py")
OUT = Path("research_outputs/fluid_stock_graph_v3")
CACHE = OUT / "cache"
MAX_CODES = 80
WORKERS = 2


def load_v2():
    spec = importlib.util.spec_from_file_location("fluid_graph_v2_runner", V2_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load corrected intraday runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MAX_CODES = MAX_CODES
    module.OUT = OUT
    module.CACHE = CACHE
    return module


def login_with_retry(attempts: int = 8):
    last = None
    for attempt in range(attempts):
        result = bs.login()
        if result.error_code == "0":
            return result
        last = result.error_msg
        time.sleep(2.0 + attempt)
    raise RuntimeError(f"BaoStock login failed after retries: {last}")


def download_batch(codes: list[str]) -> list[dict[str, object]]:
    v2 = load_v2()
    graph = v2.load_module()
    graph.MAX_CODES = MAX_CODES
    login_with_retry()
    receipts: list[dict[str, object]] = []
    try:
        for index, code in enumerate(codes, start=1):
            daily = graph.download_daily(code)
            hourly = graph.download_hourly(code)
            if daily.empty or hourly.empty:
                receipts.append({"code": code, "ok": False, "error": "empty_daily_or_hourly"})
            else:
                CACHE.mkdir(parents=True, exist_ok=True)
                stem = code.replace(".", "_")
                daily.to_pickle(CACHE / f"{stem}.daily.pkl")
                hourly.to_pickle(CACHE / f"{stem}.hourly.pkl")
                receipts.append(
                    {
                        "code": code,
                        "ok": True,
                        "daily_rows": int(len(daily)),
                        "hourly_rows": int(len(hourly)),
                        "hourly_min": str(hourly["date"].min().date()),
                        "hourly_max": str(hourly["date"].max().date()),
                    }
                )
            if index % 5 == 0:
                print(f"intraday batch progress={index}/{len(codes)} usable={sum(bool(x.get('ok')) for x in receipts)}", flush=True)
    finally:
        bs.logout()
    return receipts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    v2 = load_v2()
    graph = v2.load_module()
    graph.MAX_CODES = MAX_CODES
    login_with_retry()
    try:
        snapshots, membership_meta = graph.load_membership_snapshots()
    finally:
        bs.logout()
    counts = Counter(code for members in snapshots.values() for code in members)
    codes = [code for code, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_CODES]]
    chunks = [codes[index::WORKERS] for index in range(WORKERS)]
    receipts: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(download_batch, chunk) for chunk in chunks]
        for future in as_completed(futures):
            receipts.extend(future.result())
    with (OUT / "persistent_session_download_receipt.json").open("w", encoding="utf-8") as handle:
        json.dump(receipts, handle, ensure_ascii=False, indent=2)
    usable = sum(bool(row.get("ok")) for row in receipts)
    if usable < 60:
        raise RuntimeError(f"only {usable} intraday series available")

    def cached_daily(code: str) -> pd.DataFrame:
        path = CACHE / f"{code.replace('.', '_')}.daily.pkl"
        return pd.read_pickle(path) if path.exists() else pd.DataFrame()

    def cached_hourly(code: str) -> pd.DataFrame:
        path = CACHE / f"{code.replace('.', '_')}.hourly.pkl"
        return pd.read_pickle(path) if path.exists() else pd.DataFrame()

    graph = v2.load_module()
    graph.MAX_CODES = MAX_CODES
    graph.OUT = OUT
    graph.load_membership_snapshots = lambda: (snapshots, membership_meta)
    graph.download_daily = cached_daily
    graph.download_hourly = cached_hourly
    graph.main()

    summary_path = OUT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["splits"] = {
            "development": "2020-01-02 to 2021-12-31",
            "validation": "2022-01-01 to 2023-12-31",
            "later_history": "2024-01-01 to 2026-11-17",
        }
        summary["download_method"] = {
            "persistent_sessions": WORKERS,
            "requested_codes": MAX_CODES,
            "successful_codes": usable,
            "reason": "avoid repeated-login failure while retaining a cross-section above the 35-name IC minimum",
        }
        summary["data_availability_correction"] = {
            "hourly_start": "2020-01-02",
            "probe_commit": "90371649e10a519d32bb0508939b93ee4f08d820",
            "old_2017_2019_development_run_invalidated": True,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
