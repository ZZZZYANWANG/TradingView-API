from __future__ import annotations

import json
import traceback

import numpy as np
import pandas as pd

import run_regime_constitutive_round24 as experiment


def fit_model(panel: pd.DataFrame, parent: str, horizon: int, start_year: int, end_year: int) -> dict[str, object]:
    frame = experiment.r22.issue_frame(panel, parent, horizon, start_year, end_year)
    target = f"y_{horizon}"
    x, columns = experiment.design_matrix(frame, parent)
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    symbols = frame["symbol"].astype(str).to_numpy()

    asset_scale = (
        frame.assign(__target=y)
        .groupby("symbol", sort=False)["__target"]
        .std(ddof=0)
        .replace(0.0, np.nan)
    )
    scale = np.array([asset_scale.get(symbol, np.nan) for symbol in symbols], dtype=float)
    y_scaled = y / scale
    valid = np.isfinite(y_scaled) & np.isfinite(x).all(axis=1)

    # Round24 uses non-overlapping 72h/120h issuance clocks. The original 5,000-row
    # minimum was a frequency mismatch; 300 still leaves far more observations than
    # the nine frozen interaction terms and applies unchanged to both development
    # cross-fits and the final 2021-2022 fit.
    if int(valid.sum()) < 300:
        raise RuntimeError(
            f"insufficient non-overlapping development observations for {parent}: {int(valid.sum())}"
        )

    x_train = x[valid]
    y_train = y_scaled[valid]
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std = np.where(std > 1.0e-8, std, 1.0)
    x_std = (x_train - mean) / std
    y_mean = float(np.nanmean(y_train))
    beta = np.linalg.solve(
        x_std.T @ x_std + experiment.RIDGE_ALPHA * np.eye(x_std.shape[1]),
        x_std.T @ (y_train - y_mean),
    )
    raw_prediction = y_mean + x_std @ beta
    prediction_scale = float(np.nanstd(raw_prediction))
    if not np.isfinite(prediction_scale) or prediction_scale <= 1.0e-8:
        prediction_scale = 1.0

    return {
        "parent": parent,
        "horizon": int(horizon),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "columns": columns,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "beta": beta.tolist(),
        "y_mean": y_mean,
        "prediction_scale": prediction_scale,
        "n_train": int(valid.sum()),
        "minimum_observations": 300,
        "fix": "align minimum sample count with non-overlapping issuance clock",
    }


experiment.fit_model = fit_model

if __name__ == "__main__":
    try:
        experiment.main()
    except Exception as exc:  # noqa: BLE001
        failure = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        experiment.write_json(experiment.OUT / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
