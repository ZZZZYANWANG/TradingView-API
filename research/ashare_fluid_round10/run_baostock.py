from __future__ import annotations

import importlib.util
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import baostock as bs
import pandas as pd

SCRIPT = Path("research/ashare_fluid_round10/run.py")
START_DATE = "2018-01-01"
END_DATE = "2026-11-15"
MAX_CODES = 180
WORKERS = 10


def load_module():
    spec = importlib.util.spec_from_file_location("ashare_cascade_core", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import cascade research core")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(result: Any) -> pd.DataFrame:
    values: list[list[str]] = []
    if result.error_code != "0":
        return pd.DataFrame()
    while result.next():
        values.append(result.get_row_data())
    return pd.DataFrame(values, columns=result.fields)


def query_snapshot(stamp: pd.Timestamp) -> list[str]:
    for lag in range(8):
        date = (stamp - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
        try:
            frame = rows(bs.query_hs300_stocks(date=date))
            if len(frame) >= 150 and "code" in frame:
                return sorted(frame["code"].astype(str).unique())
        except Exception:
            continue
    return []


def membership_snapshots() -> tuple[dict[pd.Timestamp, set[str]], dict[str, Any]]:
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    try:
        snapshots: dict[pd.Timestamp, set[str]] = {}
        for year in range(2018, 2027):
            for month in (1, 4, 7, 10):
                stamp = pd.Timestamp(year=year, month=month, day=15)
                codes = query_snapshot(stamp)
                if codes:
                    snapshots[stamp] = set(codes)
                    print(f"snapshot {stamp.date()} members={len(codes)}", flush=True)
        current = rows(bs.query_hs300_stocks())
        if len(current) >= 150:
            snapshots[pd.Timestamp(END_DATE)] = set(current["code"].astype(str))
    finally:
        bs.logout()
    if not snapshots:
        raise RuntimeError("no CSI300 membership snapshots")
    counts = Counter(code for members in snapshots.values() for code in members)
    selected = [code for code, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_CODES]]
    selected_set = set(selected)
    clipped = {date: members & selected_set for date, members in snapshots.items()}
    meta = {
        "snapshot_count": len(clipped),
        "snapshot_min": str(min(clipped).date()),
        "snapshot_max": str(max(clipped).date()),
        "union_members": len(counts),
        "selected_codes": len(selected),
        "selection_rule": "top historical CSI300 quarterly membership frequency; code tiebreak",
        "snapshot_sizes_after_selection": {str(date.date()): len(members) for date, members in clipped.items()},
    }
    return clipped, meta


def download_one(code: str) -> tuple[str, pd.DataFrame | None, str | None]:
    login = bs.login()
    if login.error_code != "0":
        return code, None, login.error_msg
    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus,isST"
    error = ""
    try:
        for attempt in range(3):
            try:
                result = bs.query_history_k_data_plus(
                    code,
                    fields,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    frequency="d",
                    adjustflag="2",
                )
                frame = rows(result)
                if not frame.empty:
                    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                    numeric = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "tradestatus", "isST"]
                    for column in numeric:
                        frame[column] = pd.to_numeric(frame[column], errors="coerce")
                    frame = frame.dropna(subset=["date", "code", "open", "close", "high", "low", "amount"])
                    return code, frame, None
                error = result.error_msg
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}:{exc}"
            time.sleep(0.5 * (attempt + 1))
    finally:
        bs.logout()
    return code, None, error or "empty"


def point_in_time_filter(panel: pd.DataFrame, snapshots: dict[pd.Timestamp, set[str]]) -> pd.DataFrame:
    dates = sorted(snapshots)
    pieces: list[pd.DataFrame] = []
    for index, start in enumerate(dates):
        stop = dates[index + 1] if index + 1 < len(dates) else pd.Timestamp(END_DATE) + pd.Timedelta(days=1)
        members = snapshots[start]
        if not members:
            continue
        piece = panel.loc[
            panel["date"].ge(start)
            & panel["date"].lt(stop)
            & panel["code"].isin(members)
        ]
        if not piece.empty:
            pieces.append(piece)
    if not pieces:
        raise RuntimeError("point-in-time membership filter produced no rows")
    return pd.concat(pieces, ignore_index=True)


def load_baostock() -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshots, membership_meta = membership_snapshots()
    selected = sorted(set().union(*snapshots.values()))
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, code): code for code in selected}
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
                print(f"daily downloads={index}/{len(selected)} usable={len(frames)}", flush=True)
    if len(frames) < 100:
        raise RuntimeError(f"insufficient downloaded histories: {len(frames)}")
    panel = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    panel = panel.loc[
        panel["tradestatus"].eq(1)
        & panel["isST"].eq(0)
        & panel[["open", "high", "low", "close", "amount"]].gt(0).all(axis=1)
    ].copy()
    panel = point_in_time_filter(panel, snapshots)
    panel["market"] = panel["code"].str.slice(0, 2).str.upper()
    panel["name"] = panel["code"]
    panel["turnover"] = panel["turn"]
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    by_market = panel.groupby("market")["code"].nunique().to_dict()
    audit = {
        "source": "Baostock adjusted daily OHLCV and historical CSI300 constituent snapshots",
        "selection": membership_meta["selection_rule"],
        "point_in_time_membership": "quarterly snapshots forward-filled until the next snapshot",
        "membership": membership_meta,
        "requested": len(selected),
        "usable": panel["code"].nunique(),
        "usable_by_market": by_market,
        "rows": len(panel),
        "min_date": str(panel["date"].min().date()),
        "max_date": str(panel["date"].max().date()),
        "errors": errors,
        "limitations": [
            "quarterly rather than daily constituent snapshots",
            "selected code set uses historical membership frequency over the full sample",
            "daily replication does not replace the local one-minute path calculation",
        ],
    }
    return panel, audit


def main() -> None:
    module = load_module()
    module.load = load_baostock
    module.main()


if __name__ == "__main__":
    main()
