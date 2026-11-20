from __future__ import annotations

import json
import math

import baostock as bs
import numpy as np
import pandas as pd

import run_research_v7_timing_congestion as timing

research = timing.research

MARKET_BASE = [
    "response_z", "gap_z", "intraday_z", "range_z", "volume_shock", "turn_shock",
    "mom5", "mom20", "mom60", "vol20",
]
FINANCIAL_BLOCK = [
    "z_profit", "z_revenue", "z_roe", "z_cashflow", "z_margin", "z_eps",
    "source", "source_coherence", "source_quality",
]
TIMING_BLOCK = [
    "report_latency_z", "revision_latency_z", "industry_event_count", "market_event_count",
    "industry_event_share", "industry_congestion", "market_congestion",
]
BLOCKS = {
    "financial_disclosure_source_block": FINANCIAL_BLOCK,
    "disclosure_timing_congestion_block": TIMING_BLOCK,
}


def fit_block(frame: pd.DataFrame, name: str, block: list[str], target: str, exit_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy().sort_values("issue_date")
    years = data["issue_date"].dt.year.to_numpy()
    y = data[target].to_numpy(float)
    exit_dates = pd.to_datetime(data[exit_col]).to_numpy(dtype="datetime64[ns]")
    x0 = data[MARKET_BASE].to_numpy(float)
    xb = data[block].to_numpy(float)
    predictions = []
    for year in sorted(set(years)):
        if year < 2020:
            continue
        test = years == year
        first = np.datetime64(pd.Timestamp(year=year, month=1, day=1))
        train = (exit_dates < first) & np.isfinite(y)
        if train.sum() < research.CONTRACT.minimum_model_rows:
            continue
        for model in ("ridge", "hgb"):
            p0 = research.fit_predict(x0[train], y[train], x0[test], model)
            p1 = research.fit_predict(
                np.column_stack([x0[train], xb[train]]), y[train],
                np.column_stack([x0[test], xb[test]]), model,
            )
            part = data.loc[test, ["issue_date", "code", target]].copy()
            part["block"] = name
            part["model"] = model
            part["base_prediction"] = p0
            part["aug_prediction"] = p1
            predictions.append(part)
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    daily_parts = []
    horizon = target.removeprefix("return_")
    for stage in ("validation", "historical_holdout", "recent_extension"):
        start, end = research.STAGES[stage]
        stage_pred = pred[pred["issue_date"].between(start, end, inclusive="left")]
        for model, group in stage_pred.groupby("model"):
            valid = group[[target, "base_prediction", "aug_prediction", "issue_date", "code"]].dropna()
            if valid.empty:
                continue
            valid = valid.copy()
            valid["loss0"] = (valid[target] - valid["base_prediction"]) ** 2
            valid["loss1"] = (valid[target] - valid["aug_prediction"]) ** 2
            daily = valid.groupby("issue_date")[["loss0", "loss1"]].mean()
            daily["delta"] = daily["loss0"] - daily["loss1"]
            mean0 = daily["loss0"].mean()
            estimate = daily["delta"].mean() / mean0 if mean0 > 0 else np.nan
            influence = (daily["delta"] - estimate * daily["loss0"]) / mean0
            influence = influence - influence.mean()
            n = len(influence)
            if n >= 2 and np.isfinite(estimate):
                indices = research.block_indices(n, research.BLOCK, research.BOOTSTRAPS)
                deviations = influence.to_numpy()[indices].mean(axis=1)
                p_two = (1 + np.sum(np.abs(deviations) >= abs(estimate))) / (research.BOOTSTRAPS + 1)
                low, high = np.quantile(deviations, [0.025, 0.975])
                ci_low, ci_high = estimate - high, estimate - low
            else:
                p_two = ci_low = ci_high = np.nan
            ic0 = research.daily_ic(valid, "base_prediction", target).mean()
            ic1 = research.daily_ic(valid, "aug_prediction", target).mean()
            rows.append({
                "block": name, "target": target, "horizon": horizon, "stage": stage, "model": model,
                "relative_mse_gain": estimate, "ci_low": ci_low, "ci_high": ci_high,
                "p_two_sided": p_two, "base_rankic": ic0, "aug_rankic": ic1,
                "rankic_increment": ic1 - ic0, "rows": len(valid), "dates": len(daily),
            })
            day = daily.reset_index().rename(columns={"issue_date": "date"})
            day["block"] = name
            day["target"] = target
            day["stage"] = stage
            day["model"] = model
            day["estimate"] = estimate
            day["influence"] = influence.to_numpy()
            daily_parts.append(day)
    pred.to_parquet(research.OUT / f"block_predictions_{name}_{target}.parquet", index=False, compression="zstd")
    return pd.DataFrame(rows), pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()


def run_blocks(panel: pd.DataFrame) -> dict:
    targets = {"1d": "return_1d", "5d": "return_5d", "20d": "return_20d"}
    metrics = []
    daily = []
    for horizon, target in targets.items():
        if target not in panel or panel[target].notna().sum() < 100:
            continue
        exit_col = "exit_date_1d" if horizon == "1d" else f"exit_date_{horizon}"
        for name, columns in BLOCKS.items():
            rows, days = fit_block(panel, name, columns, target, exit_col)
            if len(rows):
                metrics.append(rows)
            if len(days):
                daily.append(days)
            print("BLOCK_EVALUATED", name, horizon, flush=True)
    metric = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    daily_frame = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()
    metric.to_csv(research.OUT / "information_block_increment.csv", index=False)
    if len(daily_frame):
        daily_frame.to_parquet(research.OUT / "information_block_daily.parquet", index=False, compression="zstd")
    admissions = []
    for name in BLOCKS:
        for horizon, target in targets.items():
            subset = metric[(metric.block == name) & (metric.target == target) & (metric.model == "hgb")]
            if subset.empty:
                continue
            stage = {row.stage: row for row in subset.itertuples()}
            required = ("validation", "historical_holdout", "recent_extension")
            passed = all(
                s in stage and stage[s].relative_mse_gain > 0 and
                stage[s].rankic_increment > 0 and stage[s].p_two_sided <= 0.10
                for s in required
            )
            admissions.append({
                "block": name, "horizon": horizon, "target": target,
                "passes_all_stages": passed, "use": "model_input_component",
                "default_enabled": False, "independent_alpha": False,
                "fresh_forward_required": True,
            })
    admission = pd.DataFrame(admissions)
    admission.to_csv(research.OUT / "information_block_admission.csv", index=False)
    return {"metrics": metric, "admission": admission}


_original_evaluate = research.evaluate


def evaluate(panel: pd.DataFrame) -> dict:
    result = _original_evaluate(panel)
    result["information_blocks"] = run_blocks(panel)
    return result


_original_admission = research.admission


def admission(results: dict) -> pd.DataFrame:
    factor = _original_admission(results)
    block = results["information_blocks"]["admission"].copy()
    block["admitted_historical_factor"] = False
    block["admitted_model_component"] = block["passes_all_stages"]
    factor["admitted_model_component"] = False
    combined = pd.concat([factor, block], ignore_index=True, sort=False)
    combined.to_csv(research.OUT / "combined_admission.csv", index=False)
    return combined


research.evaluate = evaluate
research.admission = admission


if __name__ == "__main__":
    research.main()
