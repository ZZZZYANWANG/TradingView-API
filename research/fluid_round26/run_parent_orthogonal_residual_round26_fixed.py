from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_parent_orthogonal_residual_round26 as r26  # noqa: E402


def fit_parent_models_sparse(
    panel: pd.DataFrame,
    spec: dict[str, Any],
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    parent = str(spec["parent"])
    horizon = int(spec["horizon"])
    target = f"y_{horizon}"
    columns = r26.parent_columns(panel, parent)
    models: dict[str, Any] = {}
    for symbol, group in panel.groupby("symbol", sort=False):
        frame = group.loc[(group["time"].dt.year >= start_year) & (group["time"].dt.year <= end_year)]
        frame = r26.r22.issue_frame(frame, parent, horizon, start_year, end_year)
        local = frame[columns + [target]].apply(pd.to_numeric, errors="coerce").dropna()
        # A 72h/120h non-overlapping issue clock yields roughly 70-120 rows per
        # asset-year. Ridge is the declared estimator and remains identified
        # under p>n; require a fixed minimum that is appropriate for this clock.
        if len(local) < 35:
            continue
        models[str(symbol)] = r26.fit_ridge(
            local[columns].to_numpy(float),
            local[target].to_numpy(float),
            r26.PARENT_ALPHA,
        )
    return {"columns": columns, "models": models, "horizon": horizon, "parent": parent}


r26.fit_parent_models = fit_parent_models_sparse


if __name__ == "__main__":
    try:
        r26.main()
    except Exception as exc:  # noqa: BLE001
        failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        r26.write_json(r26.OUT / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
