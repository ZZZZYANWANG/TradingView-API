from __future__ import annotations

import math

import numpy as np
import pandas as pd

import run_research_v8_information_blocks as blocks

research = blocks.research


def strict_model_increment(frame: pd.DataFrame, candidate: str, target: str, exit_col: str) -> pd.DataFrame:
    data = frame.copy().sort_values("issue_date")
    years = data["issue_date"].dt.year.to_numpy()
    target_values = data[target].to_numpy(float)
    exact_exit = pd.to_datetime(data[exit_col])
    exit_dates = exact_exit.to_numpy(dtype="datetime64[ns]")
    base = data[research.BASE_FEATURES].to_numpy(float)
    extra = data[[candidate]].to_numpy(float)
    predictions = []
    for year in sorted(set(years)):
        if year < 2020:
            continue
        test = years == year
        if not test.any():
            continue
        first = np.datetime64(pd.Timestamp(year=year, month=1, day=1))
        train = (exit_dates < first) & np.isfinite(target_values)
        if train.sum() < research.CONTRACT.minimum_model_rows:
            continue
        for model in ("ridge", "hgb"):
            p0 = research.fit_predict(base[train], target_values[train], base[test], model)
            p1 = research.fit_predict(
                np.column_stack([base[train], extra[train]]), target_values[train],
                np.column_stack([base[test], extra[test]]), model,
            )
            part = data.loc[test, ["issue_date", "code", target, exit_col]].copy()
            part["candidate"] = candidate
            part["model"] = model
            part["base_prediction"] = p0
            part["aug_prediction"] = p1
            predictions.append(part)
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if pred.empty:
        return pd.DataFrame()
    rows = []
    for stage in ("validation", "historical_holdout", "recent_extension"):
        start, end = research.STAGES[stage]
        d = pred[
            pred["issue_date"].ge(start) & pred["issue_date"].lt(end)
            & pd.to_datetime(pred[exit_col]).notna() & pd.to_datetime(pred[exit_col]).lt(end)
        ]
        for model, group in d.groupby("model"):
            valid = group[[target, "base_prediction", "aug_prediction", "issue_date", "code"]].dropna()
            if valid.empty:
                continue
            loss0 = (valid[target] - valid["base_prediction"]) ** 2
            loss1 = (valid[target] - valid["aug_prediction"]) ** 2
            daily = pd.DataFrame({
                "date": valid["issue_date"].to_numpy(), "loss0": loss0.to_numpy(), "loss1": loss1.to_numpy()
            }).groupby("date").mean()
            mse0, mse1 = daily.loss0.mean(), daily.loss1.mean()
            ic0 = research.daily_ic(valid, "base_prediction", target).mean()
            ic1 = research.daily_ic(valid, "aug_prediction", target).mean()
            rows.append({
                "candidate": candidate, "target": target, "stage": stage, "model": model,
                "relative_mse_gain": (mse0 - mse1) / mse0 if mse0 > 0 else np.nan,
                "base_rankic": ic0, "aug_rankic": ic1, "rankic_increment": ic1 - ic0,
                "rows": len(valid), "dates": len(daily), "exact_exit_stage_filter": True,
            })
    pred.to_parquet(research.OUT / f"predictions_{candidate}_{target}.parquet", index=False, compression="zstd")
    return pd.DataFrame(rows)


def strict_account_diagnostic(frame: pd.DataFrame, candidate: str, target: str) -> pd.DataFrame:
    horizon = target.removeprefix("return_")
    rows = []
    for stage in ("validation", "historical_holdout", "recent_extension"):
        data = frame[research.stage_mask(frame, stage, horizon)].copy()
        cohorts = []
        for date, group in data.groupby("issue_date"):
            valid = group[[candidate, target, "market_ret"]].dropna().sort_values(candidate)
            if len(valid) < 10:
                continue
            k = max(2, int(math.floor(len(valid) * 0.2)))
            bottom, top = valid.head(k), valid.tail(k)
            cohorts.append((
                date,
                top[target].mean() - bottom[target].mean(),
                top[target].mean() - valid[target].mean(),
                len(valid),
            ))
        if not cohorts:
            continue
        cohort = pd.DataFrame(cohorts, columns=["date", "gross_ls", "gross_long_active", "n"])
        for cost in research.COSTS_BPS:
            # One unit long and one unit short, each entered and exited: four one-way notional legs.
            charge = 4.0 * cost * 1e-4
            rows.append({
                "candidate": candidate, "target": target, "stage": stage,
                "cost_per_side_bp": cost, "cohorts": len(cohort),
                "gross_ls_mean": cohort.gross_ls.mean(), "net_ls_mean": cohort.gross_ls.mean() - charge,
                "gross_long_active_mean": cohort.gross_long_active.mean(),
                "net_long_active_mean": cohort.gross_long_active.mean() - charge,
                "positive_ls_share": float((cohort.gross_ls > 0).mean()),
                "positive_long_active_share": float((cohort.gross_long_active > 0).mean()),
                "one_way_notional_legs": 4,
            })
    return pd.DataFrame(rows)


def strict_fit_block(frame: pd.DataFrame, name: str, block: list[str], target: str, exit_col: str):
    data = frame.copy().sort_values("issue_date")
    years = data.issue_date.dt.year.to_numpy()
    y = data[target].to_numpy(float)
    exact_exit = pd.to_datetime(data[exit_col])
    exit_dates = exact_exit.to_numpy(dtype="datetime64[ns]")
    x0 = data[blocks.MARKET_BASE].to_numpy(float)
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
            part = data.loc[test, ["issue_date", "code", target, exit_col]].copy()
            part["block"] = name
            part["model"] = model
            part["base_prediction"] = p0
            part["aug_prediction"] = p1
            predictions.append(part)
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows, daily_parts = [], []
    horizon = target.removeprefix("return_")
    for stage in ("validation", "historical_holdout", "recent_extension"):
        start, end = research.STAGES[stage]
        stage_pred = pred[
            pred.issue_date.ge(start) & pred.issue_date.lt(end)
            & pd.to_datetime(pred[exit_col]).notna() & pd.to_datetime(pred[exit_col]).lt(end)
        ]
        for model, group in stage_pred.groupby("model"):
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
                "exact_exit_stage_filter": True,
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


research.model_increment = strict_model_increment
research.account_diagnostic = strict_account_diagnostic
blocks.fit_block = strict_fit_block


if __name__ == "__main__":
    research.main()
