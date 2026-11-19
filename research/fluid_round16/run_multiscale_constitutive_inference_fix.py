from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_multiscale_constitutive as experiment  # noqa: E402

OUT = Path("research_outputs/fluid_multiscale_constitutive_round16_inference_fix")
OUT.mkdir(parents=True, exist_ok=True)
SEED = experiment.SEED


def nonoverlap_issue_block_bootstrap(
    series: pd.Series,
    block: int,
    repetitions: int = 10_000,
) -> dict[str, float]:
    """Bootstrap timestamp RankIC on the already non-overlapping issue grid.

    ``base.evaluate`` first samples one issue every prediction horizon.  The
    input series therefore contains non-overlapping forecast cycles.  Treating
    ``horizon / 24`` as a number of *issue observations* double-counts the
    horizon and makes the 480-hour test undefined (about 36 observations versus
    a 40-observation minimum).  Four circular issue blocks preserve slow serial
    dependence while keeping inference defined for the frozen 20-day tests.
    """
    values = np.asarray(series.dropna(), dtype=float)
    if len(values) < 16:
        return {"low": float("nan"), "high": float("nan"), "p_nonpositive": float("nan")}
    block_observations = min(4, max(2, len(values) // 8))
    rng = np.random.default_rng(SEED)
    starts = np.arange(len(values))
    means = np.empty(repetitions, dtype=float)
    offsets = np.arange(block_observations)
    for i in range(repetitions):
        sample: list[float] = []
        while len(sample) < len(values):
            start = int(rng.choice(starts))
            indices = (start + offsets) % len(values)
            sample.extend(values[indices].tolist())
        means[i] = float(np.mean(sample[: len(values)]))
    return {
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0.0)),
    }


def main() -> None:
    experiment.OUT = OUT
    experiment.base.OUT = OUT
    experiment.base.moving_block_bootstrap = nonoverlap_issue_block_bootstrap
    experiment.main()

    audit_path = OUT / "DATA_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["inference_fix"] = {
        "reason": "issue_subset already enforces one issue per horizon; the prior bootstrap interpreted horizon-days as issue observations and was undefined for 480h",
        "resampling": "circular moving blocks on timestamp-level RankIC",
        "block_observations": 4,
        "bootstrap_repetitions": 10000,
        "selection_changed": False,
        "formula_sign_horizon_changed": False,
        "holdout_policy": "2025-2026 remains closed unless the corrected full validation gate passes",
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    report_path = OUT / "REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    prefix = (
        "# Inference correction\n\n"
        "The issue grid is already spaced by the prediction horizon. The prior bootstrap "
        "used horizon-days again as a number of issue observations, so 480-hour tests "
        "could not produce a distribution. This run changes only the timestamp-level "
        "resampling unit; candidate formulas, signs, spans, horizons and validation gates "
        "remain frozen.\n\n"
    )
    report_path.write_text(prefix + report, encoding="utf-8")


if __name__ == "__main__":
    main()
