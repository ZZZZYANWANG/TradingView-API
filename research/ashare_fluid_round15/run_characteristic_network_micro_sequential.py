from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import baostock as bs
import pandas as pd

HERE = Path(__file__).resolve().parent
ROUND10 = HERE.parent / "ashare_fluid_round10"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROUND10) not in sys.path:
    sys.path.insert(0, str(ROUND10))

import run_characteristic_network as research  # noqa: E402
import run_baostock as source  # noqa: E402

source.MAX_CODES = 40


def load_micro_sequential() -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshots, membership_meta = source.membership_snapshots()
    selected = sorted(set().union(*snapshots.values()))
    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,tradestatus,isST"
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    source.login_with_retry("micro-sequential", attempts=12)
    try:
        for index, code in enumerate(selected, start=1):
            frame: pd.DataFrame | None = None
            error = ""
            for attempt in range(4):
                try:
                    result = bs.query_history_k_data_plus(
                        code,
                        fields,
                        start_date=source.START_DATE,
                        end_date=source.END_DATE,
                        frequency="d",
                        adjustflag="2",
                    )
                    candidate = source.rows(result)
                    if not candidate.empty:
                        candidate["date"] = pd.to_datetime(candidate["date"], errors="coerce")
                        numeric = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "tradestatus", "isST"]
                        for column in numeric:
                            candidate[column] = pd.to_numeric(candidate[column], errors="coerce")
                        candidate = candidate.dropna(subset=["date", "code", "open", "close", "high", "low", "amount"])
                        frame = candidate
                        break
                    error = result.error_msg
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}:{exc}"
                time.sleep(0.5 + attempt)
            if frame is not None and not frame.empty:
                frames.append(frame)
            else:
                errors.append({"code": code, "error": error or "empty"})
            if index % 10 == 0:
                print(f"micro sequential={index}/{len(selected)} usable={len(frames)}", flush=True)
    finally:
        try:
            bs.logout()
        except Exception:
            pass
    if len(frames) < 35:
        raise RuntimeError(f"insufficient sequential micro histories: {len(frames)}; errors={errors[:5]}")
    panel = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    panel = panel.loc[
        panel["tradestatus"].eq(1)
        & panel["isST"].eq(0)
        & panel[["open", "high", "low", "close", "amount"]].gt(0).all(axis=1)
    ].copy()
    panel = source.point_in_time_filter(panel, snapshots)
    panel["market"] = panel["code"].str.slice(0, 2).str.upper()
    panel["name"] = panel["code"]
    panel["turnover"] = panel["turn"]
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    audit = {
        "source": "Baostock adjusted daily OHLCV and historical CSI300 snapshots",
        "micro_screen_only": True,
        "download_mode": "single authenticated session, sequential queries",
        "requested": len(selected),
        "usable": int(panel["code"].nunique()),
        "rows": int(len(panel)),
        "min_date": str(panel["date"].min().date()),
        "max_date": str(panel["date"].max().date()),
        "membership": membership_meta,
        "errors": errors,
        "limitation": "40-code implementation screen; cannot confer factor admission",
    }
    return panel, audit


research.source.load_baostock = load_micro_sequential
research.OUT = Path("research_outputs/ashare_fluid_round15_micro_sequential")
research.OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    research.main()
