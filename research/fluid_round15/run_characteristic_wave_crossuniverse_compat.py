from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_characteristic_wave_crossuniverse as experiment  # noqa: E402

original_merge = experiment.r.merge_external


def _ns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["time"] = pd.to_datetime(result["time"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    return result


def merge_external_ns(
    klines: pd.DataFrame,
    metrics: dict[str, pd.DataFrame],
    funding: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    return original_merge(
        _ns(klines),
        {symbol: _ns(frame) for symbol, frame in metrics.items()},
        {symbol: _ns(frame) for symbol, frame in funding.items()},
    )


experiment.r.merge_external = merge_external_ns

if __name__ == "__main__":
    experiment.main()
