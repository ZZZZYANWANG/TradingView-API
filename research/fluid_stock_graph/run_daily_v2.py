from __future__ import annotations

import importlib.util
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import baostock as bs
import pandas as pd

SCRIPT = Path("research/fluid_stock_graph/run_daily.py")
GRAPH_SCRIPT = Path("research/fluid_stock_graph/run_research.py")
OUT = Path("research_outputs/fluid_stock_daily_v2")
CACHE = OUT / "cache"
MAX_CODES = 120
WORKERS = 2


def load_graph_module():
    spec = importlib.util.spec_from_file_location("fluid_graph_shared", GRAPH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load graph research module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MAX_CODES = MAX_CODES
    return module


def load_daily_module():
    spec = importlib.util.spec_from_file_location("fluid_daily_shared", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load daily research module")
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
    graph = load_graph_module()
    login_with_retry()
    receipts: list[dict[str, object]] = []
    try:
        for index, code in enumerate(codes, start=1):
            frame = graph.download_daily(code)
            if frame.empty:
                receipts.append({"code": code, "ok": False, "error": "empty_daily"})
            else:
                CACHE.mkdir(parents=True, exist_ok=True)
                frame.to_pickle(CACHE / f"{code.replace('.', '_')}.pkl")
                receipts.append({"code": code, "ok": True, "rows": int(len(frame))})
            if index % 10 == 0:
                print(f"batch pid progress={index}/{len(codes)} usable={sum(bool(x.get('ok')) for x in receipts)}", flush=True)
    finally:
        bs.logout()
    return receipts


def cached_download_one(code: str) -> dict[str, object]:
    path = CACHE / f"{code.replace('.', '_')}.pkl"
    if not path.exists():
        return {"code": code, "ok": False, "error": "cache_missing"}
    frame = pd.read_pickle(path)
    return {"code": code, "ok": True, "rows": int(len(frame))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    graph = load_graph_module()
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
    if usable < 90:
        raise RuntimeError(f"only {usable} daily series available")

    daily = load_daily_module()
    daily.download_one = cached_download_one
    daily.ProcessPoolExecutor = ThreadPoolExecutor
    daily.main()

    summary_path = OUT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["download_method"] = {
            "persistent_sessions": WORKERS,
            "requested_codes": MAX_CODES,
            "successful_codes": usable,
            "reason": "avoid BaoStock repeated-login failures observed in the high-concurrency run",
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
