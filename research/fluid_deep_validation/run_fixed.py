from __future__ import annotations

import pandas as pd

import run as research


def evaluation_rows(
    panel: pd.DataFrame,
    factor: str,
    horizon: int,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Unit-safe non-overlapping issuance for ns/us datetime dtypes."""

    mask = panel["time"] >= pd.Timestamp(start, tz="UTC")
    if end is not None:
        mask &= panel["time"] < pd.Timestamp(end, tz="UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    epoch_hours = ((panel["time"] - epoch) // pd.Timedelta(hours=1)).astype("int64")
    mask &= epoch_hours % horizon == 0
    columns = [
        "time",
        "symbol",
        factor,
        f"y_{horizon}",
        "rv24_price",
        *research.BASE_FEATURES,
    ]
    result = (
        panel.loc[mask, columns]
        .replace([float("inf"), float("-inf")], pd.NA)
        .dropna()
        .copy()
    )
    return result.rename(columns={factor: "factor", f"y_{horizon}": "y"})


research.evaluation_rows = evaluation_rows

if __name__ == "__main__":
    research.main()
