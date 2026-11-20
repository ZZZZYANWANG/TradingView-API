from __future__ import annotations

import numpy as np
import pandas as pd

import run_research as study

base = study.base

MARKET_BASE = [
    "response_z", "gap_z", "intraday_z", "range_z", "volume_shock", "turn_shock",
    "mom5", "mom20", "mom60", "vol20",
]
REVISION_BLOCK = [
    "source", "source_quality", "source_coherence", *study.COMPONENTS,
    "revision_latency_t", "revision_number",
]


def fit_block(frame: pd.DataFrame, target: str, exit_col: str) -> pd.DataFrame:
    data = frame.copy().sort_values("issue_date")
    years = data.issue_date.dt.year.to_numpy()
    y = data[target].to_numpy(float)
    exits = pd.to_datetime(data[exit_col]).to_numpy(dtype="datetime64[ns]")
    x0 = data[MARKET_BASE].to_numpy(float)
    xb = data[REVISION_BLOCK].to_numpy(float)
    predictions = []
    for year in sorted(set(years)):
        if year < 2020:
            continue
        test = years == year
        first = np.datetime64(pd.Timestamp(year=year, month=1, day=1))
        train = (exits < first) & np.isfinite(y)
        if train.sum() < base.CONTRACT.minimum_model_rows:
            continue
        for model in ("ridge", "hgb"):
            p0 = base.fit_predict(x0[train], y[train], x0[test], model)
            p1 = base.fit_predict(
                np.column_stack([x0[train], xb[train]]), y[train],
                np.column_stack([x0[test], xb[test]]), model,
            )
            part = data.loc[test, ["issue_date", "code", target, exit_col]].copy()
            part["model"] = model
            part["base_prediction"] = p0
            part["aug_prediction"] = p1
            predictions.append(part)
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    rows = []
    if pred.empty:
        return pd.DataFrame()
    for stage in ("validation", "historical_holdout", "recent_extension"):
        start, end = base.STAGES[stage]
        d = pred[
            pred.issue_date.ge(start) & pred.issue_date.lt(end)
            & pd.to_datetime(pred[exit_col]).notna() & pd.to_datetime(pred[exit_col]).lt(end)
        ]
        for model, group in d.groupby("model"):
            valid = group[[target, "base_prediction", "aug_prediction", "issue_date", "code"]].dropna().copy()
            if valid.empty:
                continue
            valid["loss0"] = (valid[target] - valid.base_prediction) ** 2
            valid["loss1"] = (valid[target] - valid.aug_prediction) ** 2
            daily = valid.groupby("issue_date")[["loss0", "loss1"]].mean()
            daily["delta"] = daily.loss0 - daily.loss1
            mean0 = daily.loss0.mean()
            estimate = daily.delta.mean() / mean0 if mean0 > 0 else np.nan
            influence = (daily.delta - estimate * daily.loss0) / mean0
            influence = influence - influence.mean()
            if len(influence) >= 2 and np.isfinite(estimate):
                indices = base.block_indices(len(influence), base.BLOCK, base.BOOTSTRAPS)
                dev = influence.to_numpy()[indices].mean(axis=1)
                p_two = (1 + np.sum(np.abs(dev) >= abs(estimate))) / (base.BOOTSTRAPS + 1)
                low, high = np.quantile(dev, [0.025, 0.975])
                ci_low, ci_high = estimate - high, estimate - low
            else:
                p_two = ci_low = ci_high = np.nan
            ic0 = base.daily_ic(valid, "base_prediction", target, min_events=5).mean()
            ic1 = base.daily_ic(valid, "aug_prediction", target, min_events=5).mean()
            rows.append({
                "block": "financial_revision_information_block", "target": target,
                "stage": stage, "model": model, "relative_mse_gain": estimate,
                "ci_low": ci_low, "ci_high": ci_high, "p_two_sided": p_two,
                "base_rankic": ic0, "aug_rankic": ic1, "rankic_increment": ic1 - ic0,
                "rows": len(valid), "dates": len(daily), "exact_exit_stage_filter": True,
            })
    return pd.DataFrame(rows)


_original_evaluate = study.evaluate


def evaluate(panel: pd.DataFrame) -> dict:
    result = _original_evaluate(panel)
    frames = []
    for horizon, target in {"1d": "return_1d", "5d": "return_5d", "20d": "return_20d"}.items():
        if target not in panel or panel[target].notna().sum() < 100:
            continue
        exit_col = "exit_date_1d" if horizon == "1d" else f"exit_date_{horizon}"
        frame = fit_block(panel, target, exit_col)
        if len(frame):
            frame["horizon"] = horizon
            frames.append(frame)
        print("REVISION_BLOCK_EVALUATED", horizon, flush=True)
    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    metrics.to_csv(study.OUT / "information_block_increment.csv", index=False)
    result["information_block"] = metrics
    return result


study.evaluate = evaluate


if __name__ == "__main__":
    study.main()
