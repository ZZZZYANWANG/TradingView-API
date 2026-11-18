from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("research/fluid_round10/run_futures_expanded_hodge.py")
spec = importlib.util.spec_from_file_location("expanded_hodge", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load expanded Hodge source")
expanded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expanded)
source = expanded.source
core = expanded.core

OUT = Path("research_outputs/fluid_cycle_relaxation_round11")
OUT.mkdir(parents=True, exist_ok=True)
SPANS = [12, 24, 48, 72, 120, 240]
HORIZON = 120
EPS = 1.0e-12


def add_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in panel.sort_values(["symbol", "time"]).groupby("symbol", sort=False):
        g = group.copy()
        cycle = g["cross_graph_cycle_share"].clip(0.0, 1.5)
        raw = -np.sign(g["p_mom24"]).replace(0.0, np.nan) * (cycle - cycle.shift(24))
        state = core.causal_z(raw, 720, 240)
        for span in SPANS:
            g[f"cross_cycle_relaxation_ewm{span}"] = state.ewm(
                span=span,
                min_periods=max(8, int(0.65 * span)),
                adjust=False,
            ).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def portfolio(frame: pd.DataFrame, factor: str, target: str, cost_bp: float) -> dict:
    previous, returns, turns, times = {}, [], [], []
    for timestamp, group in frame.groupby("time", sort=True):
        z = group[["symbol", factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < 8:
            continue
        raw = np.tanh(z[factor].to_numpy(dtype=float) / 2.0)
        raw -= np.mean(raw)
        den = float(np.sum(np.abs(raw)))
        if den < EPS:
            continue
        weights = dict(zip(z["symbol"], raw / den))
        targets = dict(zip(z["symbol"], z[target].to_numpy(dtype=float)))
        gross = float(sum(weights[s] * targets[s] for s in weights))
        names = set(previous) | set(weights)
        turnover = float(sum(abs(weights.get(s, 0.0) - previous.get(s, 0.0)) for s in names))
        returns.append(gross - cost_bp / 10000.0 * turnover)
        turns.append(turnover); times.append(timestamp); previous = weights
    values = np.asarray(returns, dtype=float)
    annual_periods = 365.25 * 24.0 / HORIZON
    if len(values) < 20:
        return {"sharpe": np.nan, "annual_log_return": np.nan, "avg_turnover": np.nan, "years": {}}
    years = pd.to_datetime(pd.Series(times), utc=True).dt.year.to_numpy()
    return {
        "sharpe": float(np.mean(values) / (np.std(values, ddof=0) + EPS) * np.sqrt(annual_periods)),
        "annual_log_return": float(np.mean(values) * annual_periods),
        "avg_turnover": float(np.mean(turns)),
        "years": {str(y): float(np.mean(values[years == y]) * annual_periods) for y in np.unique(years)},
    }


def evaluate(panel: pd.DataFrame, factor: str, train_end: int, start: int, end: int) -> dict:
    target = f"y_{HORIZON}"
    train = core.issue_subset(panel.loc[panel["time"].dt.year <= train_end].copy(), HORIZON, "futures")
    test = core.issue_subset(panel.loc[(panel["time"].dt.year >= start) & (panel["time"].dt.year <= end)].copy(), HORIZON, "futures")
    cs, values = core.cross_sectional_ic(test, factor, target, "time", 8)
    boot = core.block_bootstrap(values, 12)
    ridge = core.ridge_increment(train, test, source.BASE, factor, target, 250000)
    test["__residual"] = core.residual_values(train, test, source.BASE, factor)
    residual_cs, _ = core.cross_sectional_ic(test, "__residual", target, "time", 8)
    ts = core.time_series_ic(test, factor, target)
    corr = core.maximum_baseline_correlation(test, factor, source.BASE)
    account = portfolio(test, factor, target, 2.0)
    return {
        "factor": factor, "period": f"{start}-{end}",
        "cs_ic": cs["mean"], "positive_share": cs["positive_share"], "n_times": cs["n_times"],
        "bootstrap_low": boot["low"], "bootstrap_high": boot["high"], "p_nonpositive": boot["p_nonpositive"],
        "ridge_gain": ridge["gain"], "residual_cs_ic": residual_cs["mean"],
        "ts_ic_diagnostic": ts["mean"], "ts_positive_share": ts["positive_share"],
        "max_baseline_corr": corr["max_abs"], "max_corr_feature": corr["feature"],
        "net_sharpe_2bp": account["sharpe"], "annual_log_return_2bp": account["annual_log_return"],
        "avg_turnover": account["avg_turnover"], "years": json.dumps(account["years"], sort_keys=True),
    }


try:
    core.FUTURES_SYMBOLS = expanded.EXPANDED_SYMBOLS
    raw, metadata = core.load_futures()
    pieces = []
    for symbol, frame in raw.groupby("symbol", sort=False):
        z = core.compute_futures_symbol(frame); z["symbol"] = symbol; pieces.append(z)
    panel = add_candidates(source.add_graph_states(pd.concat(pieces, ignore_index=True)))

    selection_rows = []
    for span in SPANS:
        factor = f"cross_cycle_relaxation_ewm{span}"
        development = evaluate(panel, factor, 2021, 2022, 2022)
        validation = evaluate(panel, factor, 2022, 2023, 2024)
        years = json.loads(validation["years"])
        stable_years = sum(1 for value in years.values() if value > 0.0)
        score = (
            validation["cs_ic"]
            + 0.80 * validation["residual_cs_ic"]
            + 10.0 * validation["ridge_gain"]
            + 0.004 * np.clip(validation["net_sharpe_2bp"], -2.0, 2.0)
            + 0.003 * stable_years
            - 0.003 * validation["avg_turnover"]
        )
        selection_rows.append({
            "span": span,
            "factor": factor,
            "development_cs_ic": development["cs_ic"],
            "development_sharpe_2bp": development["net_sharpe_2bp"],
            "validation_cs_ic": validation["cs_ic"],
            "validation_residual_cs_ic": validation["residual_cs_ic"],
            "validation_ridge_gain": validation["ridge_gain"],
            "validation_sharpe_2bp": validation["net_sharpe_2bp"],
            "validation_turnover": validation["avg_turnover"],
            "validation_bootstrap_low": validation["bootstrap_low"],
            "positive_validation_years": stable_years,
            "selection_score": score,
        })
    selection = pd.DataFrame(selection_rows).sort_values("selection_score", ascending=False)
    eligible = selection.loc[
        (selection["development_cs_ic"] > 0.0)
        & (selection["validation_cs_ic"] > 0.012)
        & (selection["validation_residual_cs_ic"] > 0.006)
        & (selection["validation_ridge_gain"] > 0.0)
        & (selection["validation_sharpe_2bp"] > 0.20)
        & (selection["positive_validation_years"] >= 1)
    ]
    chosen = (eligible if len(eligible) else selection).iloc[0]
    selected_factor = str(chosen["factor"])
    confirmation = evaluate(panel, selected_factor, 2024, 2025, 2025)
    shadow = evaluate(panel, selected_factor, 2025, 2026, 2026)
    final_pass = bool(
        confirmation["cs_ic"] > 0.010
        and confirmation["residual_cs_ic"] > 0.005
        and confirmation["ridge_gain"] > 0.0
        and confirmation["bootstrap_low"] > -0.005
        and confirmation["net_sharpe_2bp"] > 0.20
        and confirmation["max_baseline_corr"] < 0.80
        and shadow["cs_ic"] > 0.0
        and shadow["residual_cs_ic"] > 0.0
        and shadow["ridge_gain"] > -0.0002
        and shadow["net_sharpe_2bp"] > 0.0
    )
    selection.to_csv(OUT / "RELAXATION_SELECTION.csv", index=False)
    pd.DataFrame([confirmation, shadow]).to_csv(OUT / "HOLDOUT_RESULTS.csv", index=False)
    summary = {
        "completed_utc": str(pd.Timestamp.now(tz="UTC")),
        "data": metadata,
        "mechanism": "graph-circulation unwind with a frozen relaxation-memory timescale",
        "candidate_spans": SPANS,
        "selection_period": "2021-2024 only",
        "selected_factor": selected_factor,
        "selected_span": int(chosen["span"]),
        "confirmation_2025": confirmation,
        "shadow_2026": shadow,
        "final_pass": final_pass,
        "governance": {
            "2025_and_2026_not_used_for_relaxation_selection": True,
            "one_canonical_timescale_only": True,
            "strong_baseline_contains_graph_parent_states": True,
            "same_information_family": "cross_cycle_unwind",
            "cost_bp_one_way": 2.0,
        },
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "PROMOTED.json").write_text(json.dumps(core.jsonable([summary] if final_pass else []), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "REPORT.md").write_text("\n".join([
        "# Cross-Cycle Relaxation Memory Test",
        "",
        "The relaxation timescale is selected only from 2021–2024. The factor remains one identity in the cross-cycle-unwind information family.",
        "",
        "## Timescale selection",
        "",
        selection.to_markdown(index=False),
        "",
        "## 2025 confirmation and 2026 shadow",
        "",
        pd.DataFrame([confirmation, shadow]).to_markdown(index=False),
        "",
        f"Final pass: **{final_pass}**",
    ]), encoding="utf-8")
    print(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (OUT / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print(failure["traceback"]); raise
