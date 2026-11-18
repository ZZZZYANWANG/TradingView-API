from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("research/fluid_round10/run_graph_hodge.py")
spec = importlib.util.spec_from_file_location("fluid_graph_hodge", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load graph Hodge source")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
core = source.core

OUT = Path("research_outputs/fluid_divergence_memory_round10")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1.0e-12
HORIZONS = [5, 10, 20]


def signed_run_age(values: pd.Series, cap: int = 20) -> pd.Series:
    sign = np.sign(values.to_numpy(dtype=float))
    out = np.full(len(sign), np.nan, dtype=float)
    last = 0.0
    age = 0
    for i, current in enumerate(sign):
        if not np.isfinite(current) or current == 0.0:
            last = 0.0
            age = 0
            continue
        if current == last:
            age += 1
        else:
            last = current
            age = 1
        out[i] = current * min(age, cap) / cap
    return pd.Series(out, index=values.index)


def add_memory_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in panel.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        g = group.copy()
        raw = -g["u"] * g["graph_divergence"]
        state = core.causal_z(raw, 252, 100)
        cycle = g["graph_cycle_share"].clip(0.0, 1.5)
        potential = g["graph_potential_share"].clip(0.0, 1.5)
        pressure_change = g["graph_pressure"].diff()
        low_cycle = np.exp(-2.0 * cycle)
        age = signed_run_age(state, 20)
        sign_persistence10 = np.sign(state).rolling(10, min_periods=7).mean().abs()
        sign_persistence20 = np.sign(state).rolling(20, min_periods=14).mean().abs()

        candidates = {
            "divergence_inventory_ewm10": state.ewm(span=10, min_periods=7, adjust=False).mean(),
            "divergence_inventory_ewm20": state.ewm(span=20, min_periods=14, adjust=False).mean(),
            "divergence_inventory_ewm40": state.ewm(span=40, min_periods=28, adjust=False).mean(),
            "low_cycle_inventory_ewm10": (state * low_cycle).ewm(span=10, min_periods=7, adjust=False).mean(),
            "low_cycle_inventory_ewm20": (state * low_cycle).ewm(span=20, min_periods=14, adjust=False).mean(),
            "potential_inventory_ewm10": (state * potential).ewm(span=10, min_periods=7, adjust=False).mean(),
            "potential_inventory_ewm20": (state * potential).ewm(span=20, min_periods=14, adjust=False).mean(),
            "divergence_persistence10": state.ewm(span=10, min_periods=7, adjust=False).mean() * sign_persistence10,
            "divergence_persistence20": state.ewm(span=20, min_periods=14, adjust=False).mean() * sign_persistence20,
            "divergence_event_age": age * state.abs().ewm(span=10, min_periods=7, adjust=False).mean(),
            "pressure_discharge_ewm10": (-pressure_change * potential * low_cycle).ewm(span=10, min_periods=7, adjust=False).mean(),
            "pressure_discharge_ewm20": (-pressure_change * potential * low_cycle).ewm(span=20, min_periods=14, adjust=False).mean(),
        }
        for name, values in candidates.items():
            g[name] = core.causal_z(values, 252, 100)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


CANDIDATES = [
    "divergence_inventory_ewm10", "divergence_inventory_ewm20", "divergence_inventory_ewm40",
    "low_cycle_inventory_ewm10", "low_cycle_inventory_ewm20",
    "potential_inventory_ewm10", "potential_inventory_ewm20",
    "divergence_persistence10", "divergence_persistence20", "divergence_event_age",
    "pressure_discharge_ewm10", "pressure_discharge_ewm20",
]
BASE = source.GRAPH_BASE + ["divergence_absorption_ewm5"]


def evaluate(panel: pd.DataFrame, factor: str, horizon: int, sign: int, train_end: int, start: int, end: int) -> dict:
    work = panel.copy()
    work["__factor"] = sign * work[factor]
    target = f"y_{horizon}"
    train = core.issue_subset(work.loc[work["date"].dt.year <= train_end].copy(), horizon, "stocks")
    test = core.issue_subset(work.loc[(work["date"].dt.year >= start) & (work["date"].dt.year <= end)].copy(), horizon, "stocks")
    cs, values = core.cross_sectional_ic(test, "__factor", target, "date", 20)
    boot = core.block_bootstrap(values, max(4, int(20 / horizon)))
    ridge = core.ridge_increment(train, test, BASE, "__factor", target, 180000)
    test["__residual"] = core.residual_values(train, test, BASE, "__factor")
    residual, _ = core.cross_sectional_ic(test, "__residual", target, "date", 20)
    corr = core.maximum_baseline_correlation(test, "__factor", BASE)
    account = core.rank_portfolio(test, "__factor", target, "date", 10.0, 252.0 / horizon, 20)
    return {
        "factor": factor, "horizon": horizon, "sign": sign,
        "cs_ic": cs["mean"], "positive_share": cs["positive_share"], "n_times": cs["n_times"],
        "bootstrap_low": boot["low"], "bootstrap_high": boot["high"], "p_nonpositive": boot["p_nonpositive"],
        "ridge_gain": ridge["gain"], "residual_cs_ic": residual["mean"],
        "max_baseline_corr": corr["max_abs"], "max_corr_feature": corr["feature"],
        "net_sharpe_10bp": account["sharpe"], "annual_log_return_10bp": account["annual_log_return"],
        "avg_turnover": account["avg_turnover"], "years": json.dumps(account["years"], sort_keys=True),
    }


def development(panel: pd.DataFrame, factor: str, horizon: int) -> dict:
    work = core.issue_subset(panel.loc[(panel["date"].dt.year >= 2014) & (panel["date"].dt.year <= 2020)].copy(), horizon, "stocks")
    target = f"y_{horizon}"
    cs, values = core.cross_sectional_ic(work, factor, target, "date", 20)
    sign = 1 if np.isfinite(cs["mean"]) and cs["mean"] >= 0.0 else -1
    signed = sign * values
    boot = core.block_bootstrap(signed, max(4, int(20 / horizon)))
    work["__factor"] = sign * work[factor]
    account = core.rank_portfolio(work, "__factor", target, "date", 10.0, 252.0 / horizon, 20)
    score = abs(cs["mean"]) + 0.004 * np.clip(account["sharpe"], -2.0, 2.0) + 0.20 * max(boot["low"], -0.02)
    return {"factor": factor, "horizon": horizon, "sign": sign,
            "development_cs_ic": abs(cs["mean"]) if np.isfinite(cs["mean"]) else np.nan,
            "development_bootstrap_low": boot["low"], "development_net_sharpe_10bp": account["sharpe"],
            "development_turnover": account["avg_turnover"], "development_score": score}


try:
    panel = add_memory_candidates(source.panel.copy())
    dev = pd.DataFrame([development(panel, factor, horizon) for factor in CANDIDATES for horizon in HORIZONS])
    eligible = dev.loc[(dev["development_cs_ic"] > 0.008) & (dev["development_net_sharpe_10bp"] > 0.0)].copy()
    chosen = (eligible if len(eligible) else dev).sort_values("development_score", ascending=False).iloc[0]
    selected_factor, selected_horizon, selected_sign = str(chosen["factor"]), int(chosen["horizon"]), int(chosen["sign"])
    validation = evaluate(panel, selected_factor, selected_horizon, selected_sign, 2020, 2021, 2023)
    validation_gate = bool(validation["cs_ic"] > 0.008 and validation["residual_cs_ic"] > 0.005 and validation["ridge_gain"] > 0.0 and validation["bootstrap_low"] > -0.002 and validation["net_sharpe_10bp"] > 0.10 and validation["max_baseline_corr"] < 0.80)
    if validation_gate:
        confirmation = evaluate(panel, selected_factor, selected_horizon, selected_sign, 2023, 2024, 2025)
        shadow = evaluate(panel, selected_factor, selected_horizon, selected_sign, 2025, 2026, 2026)
        final_pass = bool(confirmation["cs_ic"] > 0.006 and confirmation["residual_cs_ic"] > 0.003 and confirmation["ridge_gain"] > 0.0 and confirmation["net_sharpe_10bp"] > 0.10 and shadow["cs_ic"] > 0.0 and shadow["residual_cs_ic"] > 0.0 and shadow["ridge_gain"] > -0.0002 and shadow["net_sharpe_10bp"] > 0.0)
    else:
        confirmation = shadow = None
        final_pass = False
    dev.to_csv(OUT / "DEVELOPMENT_GRID.csv", index=False)
    pd.DataFrame([validation]).to_csv(OUT / "VALIDATION_2021_2023.csv", index=False)
    if confirmation is not None:
        pd.DataFrame([confirmation]).to_csv(OUT / "CONFIRMATION_2024_2025.csv", index=False)
        pd.DataFrame([shadow]).to_csv(OUT / "SHADOW_2026.csv", index=False)
    summary = {"completed_utc": str(pd.Timestamp.now(tz="UTC")),
               "mechanism": "low-turnover storage and hysteresis of graph-divergence absorption",
               "candidate_count": len(CANDIDATES), "selected_factor": selected_factor,
               "selected_horizon": selected_horizon, "selected_sign": selected_sign,
               "validation": validation, "validation_gate": validation_gate,
               "confirmation": confirmation, "shadow": shadow, "final_pass": final_pass,
               "governance": {"development": "2014-2020", "validation": "2021-2023", "confirmation": "2024-2025 sealed until gate", "shadow": "2026", "baseline_contains_raw_divergence_absorption": True, "fixed_universe_warning": True}}
    (OUT / "SUMMARY.json").write_text(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "PROMOTED.json").write_text(json.dumps(core.jsonable([summary] if final_pass else []), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Graph-Divergence Absorption Memory Test", "", "## Development grid", "", dev.sort_values("development_score", ascending=False).to_markdown(index=False), "", "## Frozen validation", "", pd.DataFrame([validation]).to_markdown(index=False), "", f"Validation gate: **{validation_gate}**", "", f"Final pass: **{final_pass}**"]
    if confirmation is not None:
        lines += ["", "## 2024-2025 confirmation", "", pd.DataFrame([confirmation]).to_markdown(index=False), "", "## 2026 shadow", "", pd.DataFrame([shadow]).to_markdown(index=False)]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (OUT / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print(failure["traceback"])
    raise
