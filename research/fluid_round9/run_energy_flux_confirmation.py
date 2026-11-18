from __future__ import annotations

import importlib.util
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

BASE_PATH = Path("research/fluid_round9/run_coupled_daily.py")
spec = importlib.util.spec_from_file_location("coupled_daily_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load coupled daily base")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
core = base.core

OUT = Path("research_outputs/fluid_energy_flux_confirmation")
OUT.mkdir(parents=True, exist_ok=True)
FACTOR = "energy_flux"
TARGET = "y_20"
HORIZON = 20
BASELINE = base.BASE
SEED = 20261118
RNG = np.random.default_rng(SEED)
EPS = 1.0e-12


def daily_cs_series(frame: pd.DataFrame, factor: str) -> pd.Series:
    values = {}
    for date, group in frame.groupby("date", sort=True):
        z = group[[factor, TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) >= 6:
            ic = core.safe_spearman(z[factor], z[TARGET], 6)
            if np.isfinite(ic):
                values[pd.Timestamp(date)] = ic
    return pd.Series(values, dtype=float).sort_index()


def ts_metrics(frame: pd.DataFrame, factor: str) -> dict:
    by_symbol = {
        symbol: core.safe_spearman(group[factor], group[TARGET], 30)
        for symbol, group in frame.groupby("symbol")
    }
    finite = np.array([value for value in by_symbol.values() if np.isfinite(value)])
    return {
        "mean": float(np.mean(finite)) if len(finite) else np.nan,
        "median": float(np.median(finite)) if len(finite) else np.nan,
        "positive_share": float(np.mean(finite > 0.0)) if len(finite) else np.nan,
        "n_symbols": int(len(finite)),
        "by_symbol": by_symbol,
    }


def nonoverlap_offsets(frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    rows = []
    for offset in range(HORIZON):
        sample = frame.loc[(frame["session_index"].astype(float) % HORIZON) == offset]
        cs = daily_cs_series(sample, factor)
        ts = ts_metrics(sample, factor)
        rows.append({
            "offset": offset,
            "n_dates": int(len(cs)),
            "cs_ic": float(cs.mean()) if len(cs) else np.nan,
            "ts_ic": ts["mean"],
            "ts_positive_share": ts["positive_share"],
        })
    return pd.DataFrame(rows)


def leave_one_symbol_out(frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    rows = []
    symbols = sorted(frame["symbol"].unique())
    for excluded in symbols:
        sample = frame.loc[frame["symbol"] != excluded]
        cs = daily_cs_series(sample, factor)
        rows.append({
            "excluded_symbol": excluded,
            "n_dates": int(len(cs)),
            "mean_cs_ic": float(cs.mean()) if len(cs) else np.nan,
            "positive_share": float((cs > 0.0).mean()) if len(cs) else np.nan,
        })
    return pd.DataFrame(rows)


def standardize(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    mean = train[columns].mean()
    std = train[columns].std(ddof=0).replace(0.0, 1.0)
    x_train = ((train[columns] - mean) / std).fillna(0.0).clip(-12.0, 12.0).to_numpy(float)
    x_test = ((test[columns] - mean) / std).fillna(0.0).clip(-12.0, 12.0).to_numpy(float)
    return x_train, x_test


def model_increment(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    columns = BASELINE + [FACTOR, TARGET, "date", "session_index"]
    tr = train[columns].replace([np.inf, -np.inf], np.nan).dropna(subset=[FACTOR, TARGET]).copy()
    te = test[columns].replace([np.inf, -np.inf], np.nan).dropna(subset=[FACTOR, TARGET]).copy()
    xb, xbt = standardize(tr, te, BASELINE)
    xf, xft = standardize(tr, te, [FACTOR])
    y = tr[TARGET].to_numpy(float)
    yt = te[TARGET].to_numpy(float)
    base_model = Ridge(alpha=25.0).fit(xb, y)
    plus_model = Ridge(alpha=25.0).fit(np.column_stack([xb, xf]), y)
    pred_base = base_model.predict(xbt)
    pred_plus = plus_model.predict(np.column_stack([xbt, xft]))
    err_base = (yt - pred_base) ** 2
    err_plus = (yt - pred_plus) ** 2
    te["loss_diff"] = err_base - err_plus
    daily = te.groupby("date")["loss_diff"].mean().sort_index()
    bootstrap = core.block_bootstrap(daily.to_numpy(float), HORIZON, 2000)
    offset_rows = []
    for offset in range(HORIZON):
        sample = te.loc[(te["session_index"].astype(float) % HORIZON) == offset]
        offset_rows.append({
            "offset": offset,
            "n": int(len(sample)),
            "mean_loss_diff": float(sample["loss_diff"].mean()) if len(sample) else np.nan,
        })
    metrics = {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "mse_base": float(np.mean(err_base)),
        "mse_plus": float(np.mean(err_plus)),
        "relative_gain": float((np.mean(err_base) - np.mean(err_plus)) / (np.mean(err_base) + EPS)),
        "daily_loss_diff_mean": float(daily.mean()),
        "daily_loss_diff_bootstrap": bootstrap,
    }
    return metrics, pd.DataFrame(offset_rows), te[["date", "session_index", "loss_diff"]]


def residual_factor(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    tr = train[BASELINE + [FACTOR]].replace([np.inf, -np.inf], np.nan).dropna(subset=[FACTOR])
    te = test[BASELINE + [FACTOR]].replace([np.inf, -np.inf], np.nan)
    xb, xbt = standardize(tr, te, BASELINE)
    model = Ridge(alpha=25.0).fit(xb, tr[FACTOR].to_numpy(float))
    residual = te[FACTOR].to_numpy(float) - model.predict(xbt)
    residual[~np.isfinite(te[FACTOR].to_numpy(float))] = np.nan
    return pd.Series(residual, index=te.index)


def sleeve_portfolio(full_panel: pd.DataFrame, start_year: int, end_year: int, cost_bp: float) -> dict:
    panel = full_panel.sort_values(["date", "symbol"]).copy()
    dates = pd.Index(sorted(panel["date"].unique()))
    symbols = sorted(panel["symbol"].unique())
    raw_weights = pd.DataFrame(0.0, index=dates, columns=symbols)
    for date, group in panel.groupby("date"):
        z = group[["symbol", FACTOR]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 6:
            raw_weights.loc[date] = np.nan
            continue
        ranks = z[FACTOR].rank(pct=True) - 0.5
        denom = float(ranks.abs().sum())
        if denom < EPS:
            raw_weights.loc[date] = np.nan
            continue
        raw_weights.loc[date, z["symbol"].tolist()] = (ranks / denom).to_numpy(float)
    active = raw_weights.shift(1).rolling(HORIZON, min_periods=1).mean()
    gross = active.abs().sum(axis=1).replace(0.0, np.nan)
    active = active.div(gross, axis=0).fillna(0.0)

    opens = panel.pivot(index="date", columns="symbol", values="p_open").reindex(index=dates, columns=symbols)
    next_return = np.log(opens.shift(-1) / opens)
    gross_pnl = (active * next_return).sum(axis=1, min_count=1)
    turnover = active.diff().abs().sum(axis=1)
    net = gross_pnl - cost_bp / 10000.0 * turnover
    mask = (net.index.year >= start_year) & (net.index.year <= end_year)
    net = net.loc[mask].dropna()
    gross_pnl = gross_pnl.loc[net.index]
    turnover = turnover.loc[net.index]
    years = {str(year): float(net.loc[net.index.year == year].sum()) for year in sorted(set(net.index.year))}
    return {
        "cost_bp": cost_bp,
        "n_days": int(len(net)),
        "annual_log_return": float(net.mean() * 365.0),
        "gross_annual_log_return": float(gross_pnl.mean() * 365.0),
        "annual_volatility": float(net.std(ddof=0) * math.sqrt(365.0)),
        "sharpe": float(net.mean() / (net.std(ddof=0) + EPS) * math.sqrt(365.0)),
        "avg_daily_turnover": float(turnover.mean()),
        "years": years,
    }


def period_metrics(frame: pd.DataFrame, factor: str) -> dict:
    cs = daily_cs_series(frame, factor)
    bootstrap = core.block_bootstrap(cs.to_numpy(float), HORIZON, 2000)
    ts = ts_metrics(frame, factor)
    offsets = nonoverlap_offsets(frame, factor)
    leave_one_out = leave_one_symbol_out(frame, factor)
    return {
        "daily_cs_ic": float(cs.mean()),
        "daily_cs_positive_share": float((cs > 0).mean()),
        "daily_cs_n": int(len(cs)),
        "daily_cs_bootstrap": bootstrap,
        "ts": ts,
        "offset_positive_share": float((offsets["cs_ic"] > 0).mean()),
        "offset_median_cs_ic": float(offsets["cs_ic"].median()),
        "leave_one_out_min_cs_ic": float(leave_one_out["mean_cs_ic"].min()),
        "leave_one_out_positive_share": float((leave_one_out["mean_cs_ic"] > 0).mean()),
    }


try:
    panel = base.panel.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    development = panel.loc[panel["date"].dt.year <= 2022].copy()
    validation = panel.loc[panel["date"].dt.year.between(2023, 2024)].copy()
    holdout = panel.loc[panel["date"].dt.year >= 2025].copy()
    train_for_holdout = panel.loc[panel["date"].dt.year <= 2024].copy()

    validation_metrics = period_metrics(validation, FACTOR)
    holdout_metrics = period_metrics(holdout, FACTOR)

    validation["energy_flux_residual"] = residual_factor(development, validation)
    holdout["energy_flux_residual"] = residual_factor(train_for_holdout, holdout)
    validation_residual = period_metrics(validation, "energy_flux_residual")
    holdout_residual = period_metrics(holdout, "energy_flux_residual")

    model_validation, model_validation_offsets, _ = model_increment(development, validation)
    model_holdout, model_holdout_offsets, loss_holdout = model_increment(train_for_holdout, holdout)

    validation_offsets = nonoverlap_offsets(validation, FACTOR)
    holdout_offsets = nonoverlap_offsets(holdout, FACTOR)
    validation_loo = leave_one_symbol_out(validation, FACTOR)
    holdout_loo = leave_one_symbol_out(holdout, FACTOR)

    portfolios_validation = [sleeve_portfolio(panel, 2023, 2024, bp) for bp in [0.0, 2.0, 5.0, 10.0]]
    portfolios_holdout = [sleeve_portfolio(panel, 2025, 2026, bp) for bp in [0.0, 2.0, 5.0, 10.0]]

    holdout_pass = bool(
        holdout_metrics["daily_cs_ic"] > 0.020
        and holdout_metrics["daily_cs_bootstrap"]["low"] > 0.0
        and holdout_metrics["offset_positive_share"] >= 0.65
        and holdout_metrics["leave_one_out_positive_share"] >= 0.75
        and holdout_metrics["ts"]["positive_share"] >= 0.625
        and holdout_residual["daily_cs_ic"] > 0.010
        and holdout_residual["daily_cs_bootstrap"]["low"] > 0.0
        and model_holdout["relative_gain"] > 0.0
        and model_holdout["daily_loss_diff_bootstrap"]["low"] > 0.0
        and (model_holdout_offsets["mean_loss_diff"] > 0.0).mean() >= 0.65
        and portfolios_holdout[2]["sharpe"] > 0.0
        and sum(value > 0.0 for value in portfolios_holdout[2]["years"].values()) >= 1
    )

    validation_offsets.to_csv(OUT / "VALIDATION_NONOVERLAP_OFFSETS.csv", index=False)
    holdout_offsets.to_csv(OUT / "HOLDOUT_NONOVERLAP_OFFSETS.csv", index=False)
    validation_loo.to_csv(OUT / "VALIDATION_LEAVE_ONE_OUT.csv", index=False)
    holdout_loo.to_csv(OUT / "HOLDOUT_LEAVE_ONE_OUT.csv", index=False)
    model_validation_offsets.to_csv(OUT / "VALIDATION_MODEL_OFFSET_GAINS.csv", index=False)
    model_holdout_offsets.to_csv(OUT / "HOLDOUT_MODEL_OFFSET_GAINS.csv", index=False)
    loss_holdout.to_csv(OUT / "HOLDOUT_DAILY_LOSS_DIFF.csv", index=False)
    pd.DataFrame(portfolios_validation).to_csv(OUT / "VALIDATION_PORTFOLIO_COST_SWEEP.csv", index=False)
    pd.DataFrame(portfolios_holdout).to_csv(OUT / "HOLDOUT_PORTFOLIO_COST_SWEEP.csv", index=False)

    summary = {
        "completed_utc": str(pd.Timestamp.now(tz="UTC")),
        "factor": FACTOR,
        "horizon_days": HORIZON,
        "formula": "z_prior252(-shear_z * activity_gap_z * basis_velocity_z)",
        "data": base.metadata,
        "selection_identity": {
            "development": "2021-2022",
            "selected_from_family": "interfacial_energy",
            "validation": "2023-2024",
            "holdout": "2025-2026-10, read only after formula/sign/horizon freeze",
        },
        "validation": validation_metrics,
        "validation_residual": validation_residual,
        "validation_model_increment": model_validation,
        "validation_portfolios": portfolios_validation,
        "holdout": holdout_metrics,
        "holdout_residual": holdout_residual,
        "holdout_model_increment": model_holdout,
        "holdout_portfolios": portfolios_holdout,
        "holdout_pass": holdout_pass,
        "governance": {
            "independent_alpha_candidate": holdout_pass,
            "default_enabled": False,
            "production_certified": False,
            "fresh_future_confirmation_required": True,
        },
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "PROMOTED.json").write_text(json.dumps(core.jsonable([summary] if holdout_pass else []), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Interfacial Energy Flux — Frozen Holdout Confirmation",
        "",
        f"Holdout pass: **{holdout_pass}**",
        "",
        "The formula, sign and 20-day horizon were frozen after 2021-2022 development and 2023-2024 validation.",
        "The 2025-2026 segment is used here as the first confirmation segment for this identity.",
        "",
        "## Validation", "", "```json", json.dumps(core.jsonable(validation_metrics), indent=2), "```",
        "", "## Holdout", "", "```json", json.dumps(core.jsonable(holdout_metrics), indent=2), "```",
        "", "## Holdout residual", "", "```json", json.dumps(core.jsonable(holdout_residual), indent=2), "```",
        "", "## Holdout model increment", "", "```json", json.dumps(core.jsonable(model_holdout), indent=2), "```",
        "", "## Holdout portfolios", "", pd.DataFrame(portfolios_holdout).to_markdown(index=False),
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (OUT / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print(failure["traceback"])
    raise
