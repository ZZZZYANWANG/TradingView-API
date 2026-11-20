from __future__ import annotations

import time
from pathlib import Path

import baostock as bs
import pandas as pd

import run_research_v5_daily as daily_only

research = daily_only.research


def _relogin() -> None:
    try:
        bs.logout()
    except Exception:
        pass
    time.sleep(2.0)
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"baostock login failed: {result.error_code} {result.error_msg}")


def _consume(result) -> list[list[str]]:
    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(f"baostock error {result.error_code}: {result.error_msg}")
    return rows


def _retry(call, *, label: str, attempts: int = 6):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            last = exc
            print("RETRY", label, attempt, "/", attempts, repr(exc), flush=True)
            if attempt == attempts:
                break
            _relogin()
            time.sleep(min(30.0, 4.0 * attempt))
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last


def fetch_reports() -> pd.DataFrame:
    frames = []
    for report_date in research.REPORT_ENDS:
        frame = _retry(
            lambda report_date=report_date: research.fetch_report_period(report_date),
            label=f"report:{report_date.date()}",
        )
        if len(frame):
            frames.append(frame)
        print("REPORT_DONE_RESILIENT", report_date.date(), len(frame), flush=True)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw.to_parquet(research.OUT / "raw_report_rows.parquet", index=False, compression="zstd")
    return raw


def fetch_membership() -> pd.DataFrame:
    path = research.CACHE / "membership.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows: list[dict] = []
    functions = (("HS300", "query_hs300_stocks"), ("ZZ500", "query_zz500_stocks"))
    for date in research.MEMBERSHIP_DATES:
        for label, function_name in functions:
            def one_snapshot(date=date, label=label, function_name=function_name):
                result = getattr(bs, function_name)(date)
                values = _consume(result)
                output = []
                for row in values:
                    item = dict(zip(result.fields, row))
                    item.update(snapshot_date=date, index_name=label)
                    output.append(item)
                return output
            rows.extend(_retry(one_snapshot, label=f"membership:{label}:{date}"))
        print("MEMBERSHIP_RESILIENT", date, len(rows), flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("membership download returned no observations")
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    frame.to_parquet(path, index=False, compression="zstd")
    return frame


def fetch_stock_daily(code: str, start: str = "2014-01-01", end: str = "2026-11-20") -> pd.DataFrame:
    safe = code.replace(".", "_")
    path = research.CACHE / f"daily_{safe}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus,isST"

    def download():
        result = bs.query_history_k_data_plus(
            code, fields, start_date=start, end_date=end, frequency="d", adjustflag="2"
        )
        rows = _consume(result)
        return pd.DataFrame(rows, columns=result.fields)

    frame = _retry(download, label=f"daily:{code}", attempts=5)
    if len(frame):
        frame["date"] = pd.to_datetime(frame["date"])
        for column in [
            "open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"
        ]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["tradestatus"] = pd.to_numeric(frame["tradestatus"], errors="coerce")
        frame["isST"] = pd.to_numeric(frame["isST"], errors="coerce")
    frame.to_parquet(path, index=False, compression="zstd")
    return frame


research.fetch_reports = fetch_reports
research.fetch_membership = fetch_membership
research.fetch_stock_daily = fetch_stock_daily
research.fetch_hourly_targets = daily_only.fetch_hourly_targets


if __name__ == "__main__":
    research.main()
