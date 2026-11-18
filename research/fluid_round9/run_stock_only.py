from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path

import pandas as pd

SOURCE = Path("research/fluid_round9/run.py")
spec = importlib.util.spec_from_file_location("fluid_round9_core", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load round9 core")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

OUT = Path("research_outputs/fluid_round9_stock")
OUT.mkdir(parents=True, exist_ok=True)

try:
    raw, metadata = core.load_stocks()
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
            "universe_warning": "fixed current large-cap universe; survivorship selection remains",
        },
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "PROMOTED.json").write_text(json.dumps(core.jsonable(promoted), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Stock Sector-Flow Fluid Validation",
        "",
        "## Selection", "", selection.to_markdown(index=False), "",
        "## Validation", "", validation.to_markdown(index=False), "",
        "## Conditional holdout", "", holdout.to_markdown(index=False) if len(holdout) else "No validation passer; holdout remained sealed.", "",
        f"Promoted factors: {len(promoted)}",
        "",
        "The fixed current large-cap universe creates survivorship-selection risk; a promoted object remains historical/default-off until point-in-time replication.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (OUT / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print(failure["traceback"])
    raise
