from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("research/fluid_round10/run_futures_graph_hodge.py")
spec = importlib.util.spec_from_file_location("futures_graph_hodge", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load futures graph Hodge source")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
core = source.core

OUT = Path("research_outputs/fluid_futures_cycle_unwind_round10")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1.0e-12
HORIZONS = [24, 72, 120]
SMOOTHING = [24, 72, 120]


def add_cycle_conversion_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["time", "symbol"]).copy()
    p["relative_mom24"] = p["p_mom24"] - p.groupby("time")["p_mom24"].transform("mean")
    parts = []
    candidate_names = []
    for _, group in p.sort_values(["symbol", "time"]).groupby("symbol", sort=False):
        g = group.copy()
        cycle = g["cross_graph_cycle_share"].clip(0.0, 1.5)
        potential = g["cross_graph_potential_share"].clip(0.0, 1.5)
        pressure = g["cross_graph_pressure"].clip(-12.0, 12.0)
        divergence = g["cross_graph_divergence"].clip(-12.0, 12.0)
        relative_shear = (g["shear_z"] - g["cross_graph_peer_shear"]).clip(-12.0, 12.0)
        raw = {
            "cycle_to_pressure_24": -(cycle - cycle.shift(24)) * pressure,
            "cycle_to_pressure_72": -(cycle - cycle.shift(72)) * pressure,
            "cycle_to_shear_24": -(cycle - cycle.shift(24)) * relative_shear,
            "cycle_to_shear_72": -(cycle - cycle.shift(72)) * relative_shear,
            "cycle_to_divergence_24": -(cycle - cycle.shift(24)) * divergence,
            "cycle_to_divergence_72": -(cycle - cycle.shift(72)) * divergence,
            "cycle_to_basis_24": -(cycle - cycle.shift(24)) * g["basis_z"],
            "cycle_to_relative_momentum_24": -(cycle - cycle.shift(24)) * np.sign(g["relative_mom24"]),
            "potentialization_pressure_24": ((potential - potential.shift(24)) - (cycle - cycle.shift(24))) * pressure,
            "potentialization_shear_24": ((potential - potential.shift(24)) - (cycle - cycle.shift(24))) * relative_shear,
        }
        for base, values in raw.items():
            state = core.causal_z(values, 720, 240)
            for span in SMOOTHING:
                name = f"{base}_ewm{span}"
                g[name] = state.ewm(span=span, min_periods=max(16, int(0.65 * span)), adjust=False).mean()
                if name not in candidate_names:
                    candidate_names.append(name)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    for name in candidate_names:
        mean = out.groupby("time")[name].transform("mean")
        std = out.groupby("time")[name].transform(lambda s: s.std(ddof=0))
        out[name] = ((out[name] - mean) / (std + EPS)).clip(-8.0, 8.0)
    return out, candidate_names


BASE = source.BASE + ["cross_cycle_unwind_ewm12"]


def evaluate(panel: pd.DataFrame, factor: str, horizon: int, sign: int, train_end: int, start: int, end: int) -> dict:
    target = f"y_{horizon}"
    work = panel.copy()
    work["__factor"] = sign * work[factor]
    train = core.issue_subset(work.loc[work["time"].dt.year <= train_end].copy(), horizon, "futures")
    test = core.issue_subset(work.loc[(work["time"].dt.year >= start) & (work["time"].dt.year <= end)].copy(), horizon, "futures")
    cs, values = core.cross_sectional_ic(test, "__factor", target, "time", 6)
    boot = core.block_bootstrap(values, max(12, int(120 / horizon)))
    ridge = core.ridge_increment(train, test, BASE, "__factor", target, 200000)
    test["__residual"] = core.residual_values(train, test, BASE, "__factor")
    residual, _ = core.cross_sectional_ic(test, "__residual", target, "time", 6)
    ts = core.time_series_ic(test, "__factor", target)
    corr = core.maximum_baseline_correlation(test, "__factor", BASE)
    account = source.portfolio(test, "__factor", target, 2.0, 365.25 * 24.0 / horizon)
    return {
        "factor": factor, "horizon": horizon, "sign": sign,
        "cs_ic": cs["mean"], "positive_share": cs["positive_share"], "n_times": cs["n_times"],
        "bootstrap_low": boot["low"], "bootstrap_high": boot["high"], "p_nonpositive": boot["p_nonpositive"],
        "ridge_gain": ridge["gain"], "residual_cs_ic": residual["mean"],
        "ts_ic_diagnostic": ts["mean"], "ts_positive_share": ts["positive_share"],
        "max_baseline_corr": corr["max_abs"], "max_corr_feature": corr["feature"],
        "net_sharpe_2bp": account["sharpe"], "annual_log_return_2bp": account["annual_log_return"],
        "avg_turnover": account["avg_turnover"], "years": json.dumps(account["years"], sort_keys=True),
    }


def development(panel: pd.DataFrame, factor: str, horizon: int) -> dict:
    target = f"y_{horizon}"
    dev = core.issue_subset(panel.loc[panel["time"].dt.year <= 2022].copy(), horizon, "futures")
    cs, values = core.cross_sectional_ic(dev, factor, target, "time", 6)
    sign = 1 if np.isfinite(cs["mean"]) and cs["mean"] >= 0.0 else -1
    signed = sign * values
    boot = core.block_bootstrap(signed, max(12, int(120 / horizon)))
    dev["__factor"] = sign * dev[factor]
    account = source.portfolio(dev, "__factor", target, 2.0, 365.25 * 24.0 / horizon)
    score = abs(cs["mean"]) + 0.003 * np.clip(account["sharpe"], -2.0, 2.0) + 0.15 * max(boot["low"], -0.03)
    return {"factor": factor, "horizon": horizon, "sign": sign,
            "development_cs_ic": abs(cs["mean"]) if np.isfinite(cs["mean"]) else np.nan,
            "development_bootstrap_low": boot["low"], "development_net_sharpe_2bp": account["sharpe"],
            "development_turnover": account["avg_turnover"], "development_score": score}


try:
    panel, candidates = add_cycle_conversion_candidates(source.panel.copy())
    dev = pd.DataFrame([development(panel, factor, horizon) for factor in candidates for horizon in HORIZONS])
    eligible = dev.loc[(dev["development_cs_ic"] > 0.012) & (dev["development_net_sharpe_2bp"] > 0.0)].copy()
    chosen = (eligible if len(eligible) else dev).sort_values("development_score", ascending=False).iloc[0]
    selected_factor, selected_horizon, selected_sign = str(chosen["factor"]), int(chosen["horizon"]), int(chosen["sign"])
    validation = evaluate(panel, selected_factor, selected_horizon, selected_sign, 2022, 2023, 2024)
    validation_gate = bool(validation["cs_ic"] > 0.012 and validation["residual_cs_ic"] > 0.006 and validation["ridge_gain"] > 0.0 and validation["bootstrap_low"] > -0.005 and validation["net_sharpe_2bp"] > 0.50 and validation["max_baseline_corr"] < 0.80)
    if validation_gate:
        confirmation = evaluate(panel, selected_factor, selected_horizon, selected_sign, 2024, 2025, 2025)
        shadow = evaluate(panel, selected_factor, selected_horizon, selected_sign, 2025, 2026, 2026)
        final_pass = bool(confirmation["cs_ic"] > 0.008 and confirmation["residual_cs_ic"] > 0.004 and confirmation["ridge_gain"] > 0.0 and confirmation["net_sharpe_2bp"] > 0.25 and shadow["cs_ic"] > 0.0 and shadow["residual_cs_ic"] > 0.0 and shadow["ridge_gain"] > -0.0002 and shadow["net_sharpe_2bp"] > 0.0)
    else:
        confirmation = shadow = None
        final_pass = False
    dev.to_csv(OUT / "DEVELOPMENT_GRID.csv", index=False)
    pd.DataFrame([validation]).to_csv(OUT / "VALIDATION_2023_2024.csv", index=False)
    if confirmation is not None:
        pd.DataFrame([confirmation]).to_csv(OUT / "CONFIRMATION_2025.csv", index=False)
        pd.DataFrame([shadow]).to_csv(OUT / "SHADOW_2026.csv", index=False)
    summary = {"completed_utc": str(pd.Timestamp.now(tz="UTC")),
               "mechanism": "cross-sectional relative-return reordering after graph-circulation dissipation",
               "candidate_count": len(candidates), "selected_factor": selected_factor,
               "selected_horizon": selected_horizon, "selected_sign": selected_sign,
               "validation": validation, "validation_gate": validation_gate,
               "confirmation": confirmation, "shadow": shadow, "final_pass": final_pass,
               "governance": {"development": "2021-2022", "validation": "2023-2024", "confirmation": "2025 sealed until gate", "shadow": "2026", "baseline_contains_raw_cycle_unwind": True, "cross_sectional_objective": True}}
    (OUT / "SUMMARY.json").write_text(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "PROMOTED.json").write_text(json.dumps(core.jsonable([summary] if final_pass else []), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Futures Cross-Sectional Cycle-Unwind Conversion Test", "", "## Development grid", "", dev.sort_values("development_score", ascending=False).to_markdown(index=False), "", "## Frozen 2023-2024 validation", "", pd.DataFrame([validation]).to_markdown(index=False), "", f"Validation gate: **{validation_gate}**", "", f"Final pass: **{final_pass}**"]
    if confirmation is not None:
        lines += ["", "## 2025 confirmation", "", pd.DataFrame([confirmation]).to_markdown(index=False), "", "## 2026 shadow", "", pd.DataFrame([shadow]).to_markdown(index=False)]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(core.jsonable(summary), indent=2, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    failure = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (OUT / "FAILURE.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print(failure["traceback"])
    raise
