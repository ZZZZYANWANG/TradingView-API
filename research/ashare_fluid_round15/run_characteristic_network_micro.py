from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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
source.WORKERS = 4


def load_micro() -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshots, membership_meta = source.membership_snapshots()
    selected = sorted(set().union(*snapshots.values()))
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=source.WORKERS) as pool:
        futures = {pool.submit(source.download_one, code): code for code in selected}
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
            if index % 10 == 0:
                print(f"micro downloads={index}/{len(selected)} usable={len(frames)}", flush=True)
    if len(frames) < 35:
        raise RuntimeError(f"insufficient micro histories: {len(frames)}; errors={errors[:5]}")
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


research.source.load_baostock = load_micro
research.OUT = Path("research_outputs/ashare_fluid_round15_micro")
research.OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    research.main()
